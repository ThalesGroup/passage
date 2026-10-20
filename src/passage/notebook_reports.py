"""Read-only, bounded producers shared by release and paper notebooks.

Census scans project metadata over the full core population. Tensor decoding and
exact-reference diagnostics are separate, explicitly bounded operations. Missing
measurements remain visible prerequisites; stored labels are never called exact
references merely because their filenames say A*.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

CORE_SPLITS = ("train", "calibration", "validation", "test")


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the checkout without changing the working directory."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/passage").is_dir():
            return candidate
    raise FileNotFoundError(f"PASSAGE checkout not found above {start}")


def resolve_path(value: str | Path | None, root: Path) -> Path | None:
    """Resolve declared relative inputs against the checkout, never private mounts."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_evidence_path(value: str | Path | None, root: Path) -> Path | None:
    """Locate a portable evidence bundle from a published notebook directory."""
    path = resolve_path(value, root)
    if value is None or Path(value).is_absolute():
        return path
    for ancestor in (Path.cwd(), *Path.cwd().parents):
        candidate = ancestor / value
        if candidate.is_dir():
            return candidate
    return path


def make_source(dataset_path: str | Path | None = "dataset", *, dataset_repo: str = "thalesgroup/passage",
                dataset_revision: str | None = None, project_root: Path | None = None):
    """Use a local declared root, or explicitly requested pinned remote data."""
    from passage.dataset import DatasetSource
    if project_root is None:
        try:
            project_root = find_project_root()
        except FileNotFoundError:
            project_root = Path.cwd()
    return DatasetSource(root=resolve_path(dataset_path, project_root),
                         repo=dataset_repo, revision=dataset_revision)


def _identity(source) -> dict[str, Any]:
    return source.identity() if hasattr(source, "identity") else {"mode": "provided_source", "content_verified": False}


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def generation_outcome_report(source, resolution: int) -> dict[str, Any]:
    """Read retained terminal denominators separately from exported tensor row counts."""
    root = getattr(source, "root", None)
    result = {"status": "not_available", "content_verified": False,
              "requested_scenes": 0, "requested_variants": 0,
              "scene_status_counts": {}, "variant_status_counts": {}, "batch_receipts": 0,
              "scope": "Declared release generation outcomes; full release admission requires receipt verification."}
    if root is None:
        return result
    scene_counts, variant_counts = Counter(), Counter()
    for path in sorted((Path(root) / "release-receipts").glob(f"batch-{resolution}-*.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("schema_version") != "passage-export-batch-v1" or receipt.get("resolution") != resolution:
            raise ValueError("Release generation outcome receipt identity mismatch")
        result["requested_scenes"] += receipt["requested_scenes"]
        result["requested_variants"] += receipt["requested_variants"]
        result["batch_receipts"] += 1
        scene_counts.update(receipt["scene_status_counts"])
        variant_counts.update(receipt["variant_status_counts"])
    if result["batch_receipts"]:
        result.update(status="declared", scene_status_counts=dict(scene_counts), variant_status_counts=dict(variant_counts))
    return result


def dataset_report(source, resolution: int, *, census_limit: int | None = None) -> dict[str, Any]:
    """Full four-split core census; no tensor/images or row-limit shortcuts."""
    if census_limit is not None and (not isinstance(census_limit, int) or census_limit <= 0):
        raise ValueError("census_limit must be None or positive")
    summaries = []
    for split in CORE_SPLITS:
        count, missing, mismatch = 0, 0, 0
        tiles, calibrations = Counter(), Counter()
        relief_sum, relief_count, obstacle_sum, obstacle_count = 0.0, 0, 0.0, 0
        for row in source.iter_rows(resolution, split, columns=["id", "metadata"], samples=False, limit=census_limit):
            count += 1
            meta = row.get("metadata") or {}
            missing += int(not row.get("id") or not meta)
            mismatch += int(meta.get("resolution") != resolution or meta.get("split") != split)
            tile = (meta.get("tile") or {}).get("source") or {}
            if tile.get("name"):
                tiles[tile["name"]] += 1
            calibration = meta.get("calibration") or {}
            calibrations[(calibration.get("min"), calibration.get("max"))] += 1
            crop = meta.get("crop") or {}
            low, high = _finite(crop.get("min_elevation")), _finite(crop.get("max_elevation"))
            if low is not None and high is not None:
                relief_sum += high-low
                relief_count += 1
            ratio = _finite((meta.get("obstacles") or {}).get("ratio_actual"))
            if ratio is not None:
                obstacle_sum += ratio
                obstacle_count += 1
        summaries.append({"split": split, "rows": count, "missing_id_or_metadata": missing,
            "metadata_mismatch": mismatch, "source_tile_count": len(tiles),
            "tile_counts": dict(sorted(tiles.items())),
            "calibrations": [{"min": lo, "max": hi, "rows": n} for (lo, hi), n in calibrations.items()],
            "mean_relief_m": relief_sum/relief_count if relief_count else None,
            "relief_denominator": relief_count,
            "mean_obstacle_fraction": obstacle_sum/obstacle_count if obstacle_count else None,
            "obstacle_denominator": obstacle_count,
            "qualification_status": "not_run",
            "qualification_reason": "Metadata census does not establish route legality, optimum or geographic isolation."})
    return {"schema_version": 1, "resolution": resolution, "population": "full_core" if census_limit is None else "bounded_core_metadata",
            "census_limit_per_split": census_limit,
            "source": _identity(source), "splits": summaries, "total_rows": sum(r["rows"] for r in summaries),
            "generation_outcomes": generation_outcome_report(source, resolution)}


def route_metrics_report(row: Mapping[str, Any], resolution: int, split: str) -> dict[str, Any]:
    """Re-cost each stored ordered route and compare same-objective free/obstacle paths.

    Geometry uses KD trees and includes identical paths (zero divergence). Costs
    are physical canonical costs, not ratios across distinct cost objectives.
    """
    from passage.utils import decode_tensor_blob
    from passage_eval.metrics.cost import path_cost
    from passage_eval.metrics.geometric import path_assd, path_hausdorff_p95
    from passage_eval.metrics.reliability import path_status

    meta = row["metadata"]
    if meta["resolution"] != resolution or meta["split"] != split:
        raise ValueError("Requested resolution/split disagrees with the row")
    calibration = meta["calibration"]
    low, high = float(calibration["min"]), float(calibration["max"])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid elevation calibration")
    tensor = decode_tensor_blob(bytes(row["tensor"]), (resolution, resolution, 3))
    elevation = tensor[..., 0].astype(np.float64)*(high-low)+low
    obstacles = tensor[..., 2].astype(bool)
    start = tuple(meta["markers"]["start_point"][axis] for axis in ("i", "j"))
    goal = tuple(meta["markers"]["end_point"][axis] for axis in ("i", "j"))
    records, comparisons, available = [], [], []
    for solver, config in meta["solvers"].items():
        for variant in ("free", "obstacles"):
            key = f"path_{solver}_{variant}"
            if key not in row:
                continue
            path = row[key]
            connected, violation, _ = path_status(path, start=start, goal=goal,
                obstacles=obstacles if variant == "obstacles" else np.zeros_like(obstacles))
            actual = path_cost(elevation, path, weight=float(config["weight"]), cost_model=config["cost_model"])
            stored = _finite(meta["paths"][solver][variant]["cost"])
            stored_status = meta["paths"][solver][variant].get("status") or "historical_unspecified"
            ordered = json.dumps(np.asarray(path).tolist(), separators=(",", ":")).encode()
            records.append({"route": key, "solver": solver, "cost_model": config["cost_model"],
                "task_variant": variant, "stored_status": stored_status, "vertices": len(path), "connected": bool(connected),
                "obstacle_violation": bool(violation), "legal": bool(connected and not violation and np.isfinite(actual)),
                "canonical_cost": _finite(actual), "stored_cost": _finite(stored),
                "cost_agreement": bool(stored is not None and np.isclose(actual, stored, rtol=1e-7, atol=1e-6)),
                "ordered_route_sha256": hashlib.sha256(ordered).hexdigest(),
                "stored_path_seconds": _finite((meta["paths"][solver].get("timing") or {}).get(f"path_{variant}_seconds")),
                "timing_scope": "Stored generation solver time; graph-build/complete-request timing is not inferred."})
            available.append((key, path, solver, variant, records[-1]["legal"]))
    for (a, pa, sa, va, la), (b, pb, sb, vb, lb) in combinations(available, 2):
        if sa != sb and va != vb:
            continue
        valid = la and lb
        comparisons.append({"route": a, "reference_route": b,
            "assd": _finite(path_assd([pa], [pb])[0]) if valid else None,
            "hausdorff_p95_pixels": _finite(path_hausdorff_p95([pa], [pb])[0]) if valid else None,
            "geometry_status": "admitted" if valid else "invalid_route",
            "comparison_scope": ("Same scene/objective, different obstacle conditions; descriptive divergence."
                                 if sa == sb else "Same scene/obstacle condition, different objectives; descriptive divergence.")})
    return {"schema_version": 1, "resolution": resolution, "split": split, "sample_id": row["id"],
        "calibration": {"min": low, "max": high}, "routes": records, "comparisons": comparisons,
        "reference_status": "stored_route_not_exact_revalidated",
        "counts": {"routes_requested": len(records), "routes_legal": sum(r["legal"] for r in records),
                   "stored_cost_agreement": sum(r["cost_agreement"] for r in records),
                   "stored_status_counts": dict(Counter(r["stored_status"] for r in records)),
                   "comparisons_requested": len(comparisons),
                   "comparisons_valid": sum(r["geometry_status"] == "admitted" for r in comparisons)}}


def classical_diagnostics_report(source, resolution: int, *, budget: int = 16,
                                 max_expansions: int = 10000) -> dict[str, Any]:
    """Bound exact-reference auditing; failed or budget-limited rows stay visible."""
    if budget < 0 or max_expansions <= 0:
        raise ValueError("Require nonnegative task budget and positive expansion budget")
    if budget == 0:
        return {"status": "not_run", "planned_tasks": 16, "requested_tasks": 0, "records": [],
                "reason": "Classical diagnostics explicitly disabled by classical_budget=0."}
    from passage.audit import audit_source
    receipt = audit_source(source, resolution=resolution, split="validation", limit=budget,
                           max_expansions=max_expansions, samples=False,
                           cost_models=("elevation", "slope", "energy"))
    return {"status": "admitted", "planned_tasks": 16, "requested_tasks": budget,
            "scope": "Bounded diagnostic cohort; no population or tail claim.",
            "task_ids": [r["sample_id"] for r in receipt.get("rows", [])],
            "expected_task_variants": budget * 6,
            "qualified_reference_status": ("admitted" if receipt["counts"]["qualified"] == budget * 6
                and receipt["counts"]["invalid_metadata"] == 0
                and receipt["coverage"]["audited_rows"] == budget else "not_run"),
            **receipt}


def evaluation_report(result_root: str | Path | None, resolution: int) -> dict[str, Any]:
    """Read measured records, preserving absent/unverified evidence explicitly.

    Aggregate metrics without a binding measured-request receipt are historical
    context only. This producer performs no inference, benchmark or training.
    """
    report = {"schema_version": 1, "resolution": resolution, "status": "not_run", "records": [],
        "historical_aggregates": [], "blocked_gate": "S2-S4",
        "required": resolution == 256,
        "learned_status": "not_run", "learned_records_status": "not_run",
        "required_sections": {"S2_component_comparison": "not_run", "S3_complete_request_frontier": "not_run",
                              "S4_joint_geographic_support": "not_run"},
        "reason": "Provide declared checkpoint/procedure identities, qualified task IDs and measured per-task receipts."}
    if result_root is None:
        return report
    root = Path(result_root)
    if not root.exists():
        report["reason"] = f"Result prerequisites missing: {root}"
        return report
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("resolution") == resolution:
            report["historical_aggregates"].append({"path": str(path), "support_status": "unverified",
                                                  "payload": payload})
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("dataset", {}).get("resolution") != resolution or manifest.get("series") == "warmup":
            continue
        score_path = path.parent / "score_receipt.json"
        if not score_path.exists():
            report.setdefault("pending_runs", []).append({"path": str(path), "reason": "Offline score receipt missing."})
            continue
        score = json.loads(score_path.read_text())
        admission = manifest.get("admission") or {}
        procedure = manifest.get("procedure") or {}
        checkpoints = manifest.get("checkpoint_hashes") or procedure.get("checkpoint_hashes")
        learned = procedure.get("family") != "classical"
        admitted = (score.get("schema") == "passage-offline-score-v1"
                    and score.get("qualified_reference") is True
                    and score.get("reference_policy") == "qualified_exact"
                    and admission.get("qualified_reference") is True
                    and admission.get("ordinary_input_contract") == "PlanRequest-v1"
                    and bool(manifest.get("procedure_id"))
                    and bool(manifest.get("dataset", {}).get("content_hash"))
                    and bool(manifest.get("reference_identity"))
                    and (not learned or bool(checkpoints)))
        if not admitted:
            report.setdefault("pending_runs", []).append({"path": str(path),
                "reason": "Incomplete procedure/input/checkpoint identity or unqualified reference receipt."})
            continue
        from passage_eval.execution.artifacts import read_run_artifact, sha256_file
        artifact = read_run_artifact(path.parent)
        for filename, key in (("run_manifest.json", "run_manifest_sha256"),
                              ("per_sample.npz", "per_sample_sha256"), ("metrics.json", "metrics_sha256")):
            file_path = path.parent / filename
            if not file_path.is_file() or score.get(key) != sha256_file(file_path):
                raise ValueError(f"{file_path}: score receipt hash mismatch")
        if score.get("scored_records") != len(artifact.records):
            raise ValueError(f"{score_path}: scored record count mismatch")
        report["records"].extend({**record, "procedure_id": manifest["procedure_id"],
                                  "family": procedure.get("family")} for record in artifact.records)
        scoped_learned = (learned and resolution == 256
                          and manifest["dataset"].get("split") in {"validation", "test"}
                          and manifest.get("bench_spec", {}).get("cost_model") == "elevation")
        report.setdefault("receipts", []).append({"path": str(path), "sha256": sha256_file(path),
            "procedure_id": manifest["procedure_id"], "dataset": manifest["dataset"],
            "reference_identity": manifest["reference_identity"], "checkpoint_hashes": checkpoints,
            "score_receipt_sha256": sha256_file(score_path), "learned": learned,
            "required_operating_point": scoped_learned,
            "metrics": json.loads((path.parent / "metrics.json").read_text())})
    report["task_ids"] = sorted({r["sample_id"] for r in report["records"]})
    if report["records"]:
        report["status"] = "admitted"
        report["learned_records_status"] = "admitted" if any(r["required_operating_point"] for r in report["receipts"]) else "not_run"
        report["reason"] = ("Hash-bound measured records are available. S2 component comparisons, S3 frontiers and "
                            "S4 joint geographic support require matched analysis receipts; individual runs do not close that gate.")
    else:
        report["learned_status"] = "not_run"
    return report


def write_report(report: Mapping[str, Any], output_dir: str | Path, filename: str) -> Path:
    """Persist a JSON report with NaN-free absence meanings."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    destination.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False, default=str)+"\n")
    return destination


def show_records(records, *, columns=None):
    """Display a compact table in a notebook, including empty status tables."""
    import pandas as pd
    from IPython.display import display
    frame = pd.DataFrame(records)
    if columns and not frame.empty:
        frame = frame[[c for c in columns if c in frame.columns]]
    display(frame)
    return frame


def stored_timing_report(source, resolutions, *, max_samples: int = 16, split: str = "validation",
                         sample_id: str | None = None) -> dict[str, Any]:
    """Describe bounded stored generation times; never invent wall-clock timings."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    records = []
    for resolution in resolutions:
        rows = ([source.get_row(resolution, split, sample_id=sample_id, columns=["id", "metadata"], samples=False)]
                if sample_id is not None else
                source.iter_rows(resolution, split, columns=["id", "metadata"], samples=False, limit=max_samples))
        for row in rows:
            for solver, paths in row["metadata"]["paths"].items():
                for variant in ("free", "obstacles"):
                    seconds = _finite((paths.get("timing") or {}).get(f"path_{variant}_seconds"))
                    records.append({"sample_id": row["id"], "resolution": resolution, "split": split,
                                    "solver": solver, "task_variant": variant, "stored_path_seconds": seconds})
    return {"schema_version": 1, "status": "admitted", "source": _identity(source), "records": records,
            "task_ids": sorted({f'{r["resolution"]}:{r["sample_id"]}' for r in records}),
            "measurement_status": "not_run",
            "timing_scope": "Stored generation path_free_seconds/path_obstacles_seconds only; no new timing or graph-build inference."}


def save_figure(figure, output_dir: str | Path, stem: str) -> list[str]:
    """Persist a displayed figure in both raster and publication formats."""
    from matplotlib import pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    # Backends select font embedding when saving, outside renderer contexts.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        for suffix in ("png", "pdf"):
            path = directory / f"{stem}.{suffix}"
            figure.savefig(path, dpi=140, bbox_inches="tight")
            files.append(str(path))
    return files


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def split_inventory_report(source, resolutions=(64, 128, 256, 512, 1024, 2048, 4096)) -> dict[str, Any]:
    """Global resolution inventory using full-core Parquet footer counts."""
    records = []
    for resolution in resolutions:
        counts = source.split_counts(resolution, samples=False)
        if set(counts) != set(CORE_SPLITS):
            raise ValueError("Split inventory requires train/calibration/validation/test separately")
        records.append({"resolution": resolution, **counts, "total": sum(counts.values())})
    return {"schema_version": 1, "population": "full_core_footer_counts", "records": records,
            "source": _identity(source), "qualification_status": "not_run"}
