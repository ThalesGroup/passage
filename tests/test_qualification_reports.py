# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Small receipt fixtures: no dataset scan or reference search."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


# Synthetic source identity: the consumer accepts any named source pinned to a 40-hex revision.
REPO = "fixture/passage"
PIN = "c" * 40

# The LaTeX/figure exporter lives in the companion paper repository, expected as a
# sibling checkout (``../paper``). Tests of that exporter skip when it is absent.
PAPER_FIGURE_SCRIPT = Path(__file__).resolve().parents[2] / "paper/scripts/render_qualification_figures.py"
requires_paper_checkout = pytest.mark.skipif(
    not PAPER_FIGURE_SCRIPT.is_file(),
    reason="companion paper checkout not found at ../paper/scripts/render_qualification_figures.py",
)


def load_paper_figure_script(module_name):
    spec = importlib.util.spec_from_file_location(module_name, PAPER_FIGURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def metadata_fixture(root):
    source = {"mode": "local", "root": "/fixture", "repo": REPO, "revision": PIN, "content_verified": False}
    protocol = {
        "protocol_version": "passage-metadata-qualification-v1",
        "source": source,
        "resolutions": [64],
        "splits": ["train", "validation", "calibration", "test"],
        "population": "core",
        "audit_counts": {"validation": 16, "train": 4, "calibration": 4, "test": 4},
        "reference_audit": {"max_expansions": 100000},
        "authentication_manifest": None,
        "manifest": {
            f"64/{s}": {"shards": [{"path": f"64x64/{s}/part.parquet", "rows": 1}]}
            for s in ("train", "validation", "calibration", "test")
        },
    }
    digest = hashlib.sha256(canonical(protocol)).hexdigest()
    cells = {
        f"64/{s}": {
            "status": "complete",
            "rows": 1,
            "expected_footer_rows": 1,
            "scanned_shards": 1,
            "total_shards": 1,
            "schema_consistent": True,
            "duplicate_id_rows": 0,
            "missing": {},
            "invalid": {},
            "footprints": {},
            "relief": {"bins": {"[0,200)": 1}, "missing": 0, "invalid": 0, "out_of_range": 0},
            "occupancy": {"bins": {"[0,0.05)": 1}, "missing": 0, "invalid": 0, "out_of_range": 0},
            "odd_strata": {"relief=[0,200);occupancy=[0,0.05)": 1},
        }
        for s in protocol["splits"]
    }
    write(root, "metadata/protocol.json", protocol)
    write(
        root,
        "metadata/summary.json",
        {
            "protocol_version": protocol["protocol_version"],
            "source": source,
            "protocol_sha256": digest,
            "status": "complete",
            "population_scope_complete": True,
            "cells": cells,
            "geography": {},
        },
    )
    return source


def test_missing_artifacts_are_explicit_and_renderable(tmp_path):
    from passage.qualification_reports import qualification_report, render_qualification_figures

    report = qualification_report(tmp_path)
    assert report["status"] == "not_run"
    assert report["completion"]["full_campaign"] is False
    assert report["counts"]["population_rows"] is None
    assert report["population_estimation"] is False
    assert report["missing_prerequisites"]
    outputs = render_qualification_figures(report, tmp_path / "figures")
    assert {"fig_qualification.pdf", "fig_qualification.png"} <= set(outputs)
    assert all((tmp_path / "figures" / name).stat().st_size > 0 for name in outputs)


def test_metadata_completion_never_implies_authenticated_or_qualified(tmp_path):
    from passage.qualification_reports import qualification_report, render_qualification_figures

    metadata_fixture(tmp_path)
    report = qualification_report(tmp_path, 64)
    assert report["completion"]["metadata_census"] is True
    assert report["completion"]["source_authentication"] is False
    assert report["completion"]["full_campaign"] is False
    assert report["counts"]["population_rows"] == 4
    assert report["resolutions"][0]["routes"]["qualified"] is None
    assert report["binding"]["report_sha256"]
    summary = json.loads((tmp_path / "metadata/summary.json").read_text())
    for cell in summary["cells"].values():
        cell["source_tiles"] = {"N001E002": 1}
    write(tmp_path, "metadata/summary.json", summary)
    report = qualification_report(tmp_path, 64)
    outputs = render_qualification_figures(report, tmp_path / "figures")
    assert "fig_qualification_metadata.pdf" in outputs
    assert "fig_qualification_tile_support.pdf" in outputs
    assert "fig_qualification_joint_odd.pdf" in outputs
    assert "fig_qualification_joint_odd.png" in outputs


def test_joint_odd_uses_all_core_rows_and_validates_unknown_counts(tmp_path):
    from passage.qualification_reports import qualification_report, qualification_joint_odd_data

    metadata_fixture(tmp_path)
    path = tmp_path / "metadata/summary.json"
    summary = json.loads(path.read_text())
    summary["cells"]["64/train"]["odd_strata"] = {"relief=missing;occupancy=[0,0.05)": 1}
    write(tmp_path, "metadata/summary.json", summary)
    joint = qualification_joint_odd_data(qualification_report(tmp_path))[0]
    assert joint["core_rows"] == 4
    assert joint["counts"][0][0] == 3
    assert joint["shares"][0][0] == 0.75
    assert joint["excluded_rows"] == 1 and joint["exclusion_counts"]["missing"] == 1
    assert joint["counts"][0][4] == 0
    assert sum(sum(row) for row in joint["counts"]) + joint["excluded_rows"] == joint["core_rows"]
    summary["cells"]["64/train"]["odd_strata"] = {}
    write(tmp_path, "metadata/summary.json", summary)
    with pytest.raises(ValueError, match="joint|strat"):
        qualification_report(tmp_path)


def test_joint_odd_seven_resolution_figure_is_bound_in_receipt(tmp_path):
    from passage.qualification_reports import (
        RESOLUTIONS,
        qualification_report,
        qualification_joint_odd_data,
        render_qualification_figures,
        with_report_figures,
    )

    metadata_fixture(tmp_path)
    protocol = json.loads((tmp_path / "metadata/protocol.json").read_text())
    summary = json.loads((tmp_path / "metadata/summary.json").read_text())
    prototype = summary["cells"]["64/train"]
    protocol["resolutions"] = list(RESOLUTIONS)
    protocol["manifest"] = {
        f"{side}/{split}": {"shards": [{"path": f"{side}x{side}/{split}/part.parquet", "rows": 1}]}
        for side in RESOLUTIONS
        for split in protocol["splits"]
    }
    summary["cells"] = {
        f"{side}/{split}": json.loads(json.dumps(prototype)) for side in RESOLUTIONS for split in protocol["splits"]
    }
    summary["protocol_sha256"] = hashlib.sha256(canonical(protocol)).hexdigest()
    write(tmp_path, "metadata/protocol.json", protocol)
    write(tmp_path, "metadata/summary.json", summary)
    report = qualification_report(tmp_path)
    joint = qualification_joint_odd_data(report)
    assert len(joint) == 7
    assert all(item["shares"][0][0] == 1 and item["core_rows"] == 4 for item in joint)
    directory = tmp_path / "all-figures"
    outputs = render_qualification_figures(report, directory)
    derived = with_report_figures(report, outputs, directory)
    for suffix in ("pdf", "png"):
        name = f"fig_qualification_joint_odd.{suffix}"
        assert str(directory / name) in derived["figures"]
        assert derived["figure_sha256"][name]["sha256"] == hashlib.sha256((directory / name).read_bytes()).hexdigest()


def test_unbound_or_wrong_pin_artifacts_are_rejected(tmp_path):
    from passage.qualification_reports import qualification_report

    metadata_fixture(tmp_path)
    path = tmp_path / "metadata/summary.json"
    value = json.loads(path.read_text())
    value["source"]["revision"] = "a" * 40
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source|identity|pin"):
        qualification_report(tmp_path)
    value["source"]["revision"] = PIN
    value["protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol"):
        qualification_report(tmp_path)


def test_requested_source_identity_is_enforced(tmp_path):
    from passage.qualification_reports import qualification_report

    missing = tmp_path / "missing"
    assert qualification_report(missing)["source"] == {"root": None, "repo": None, "revision": None}
    requested = {"repo": REPO, "revision": PIN}
    assert qualification_report(missing, source=requested)["source"] == {"root": None, **requested}
    metadata_fixture(tmp_path)
    assert qualification_report(tmp_path, source=requested)["source"]["revision"] == PIN
    assert qualification_report(tmp_path, source={"repo": REPO})["source"]["repo"] == REPO
    with pytest.raises(ValueError, match="requested source identity"):
        qualification_report(tmp_path, source={"repo": REPO, "revision": "a" * 40})
    with pytest.raises(ValueError, match="40-hex"):
        qualification_report(tmp_path, source={"revision": "pinned"})


def test_unpinned_source_identity_is_rejected(tmp_path):
    from passage.qualification_reports import qualification_report

    metadata_fixture(tmp_path)
    for name in ("metadata/protocol.json", "metadata/summary.json"):
        path = tmp_path / name
        value = json.loads(path.read_text())
        value["source"]["revision"] = "main"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="40-hex"):
        qualification_report(tmp_path)


def test_requested_resolution_is_validated(tmp_path):
    from passage.qualification_reports import qualification_report

    with pytest.raises(ValueError, match="resolution"):
        qualification_report(tmp_path, 63)


def route_fixture(root):
    source = metadata_fixture(root)
    path = root / "metadata/protocol.json"
    protocol = json.loads(path.read_text())
    shard = {
        "path": "64x64/validation/part.parquet",
        "rows": 1,
        "size_bytes": 10,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    protocol["manifest"]["64/validation"]["shards"] = [shard]
    write(root, "metadata/protocol.json", protocol)
    metadata_id = hashlib.sha256(canonical(protocol)).hexdigest()
    summary_path = root / "metadata/summary.json"
    metadata = json.loads(summary_path.read_text())
    metadata["protocol_sha256"] = metadata_id
    write(root, "metadata/summary.json", metadata)
    entry = {
        "resolution": 64,
        "split": "validation",
        "sample_id": "a",
        "locator": {"shard": shard["path"], "row_index": 0, "row_group": 0, "row_in_group": 0},
    }
    roster = {
        "source": source,
        "protocol_sha256": metadata_id,
        "population_estimation": False,
        "frozen_before_tensor_or_reference_audit": True,
        "scenes": [entry],
    }
    roster_id = hashlib.sha256(canonical(roster)).hexdigest()
    roster["roster_sha256"] = roster_id
    write(root, "metadata/audit_roster.json", roster)
    route = {
        "protocol_version": "passage-route-qualification-v1",
        "source": source,
        "metadata_protocol_sha256": metadata_id,
        "metadata_summary_sha256": hashlib.sha256(canonical(metadata)).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(canonical(protocol["manifest"])).hexdigest(),
        "roster_sha256": roster_id,
        "population_estimation": False,
        "authentication_manifest": None,
        "max_expansions": 100000,
        "rtol": 1e-7,
        "atol": 1e-6,
        "cost_models": ["elevation", "slope_uphill", "energy"],
        "task_variants": ["free", "obstacles"],
        "retain_failures_and_censoring": True,
    }
    route_id = hashlib.sha256(canonical(route)).hexdigest()
    write(root, "route-audit/protocol.json", route)
    row_id = hashlib.sha256(canonical({k: entry[k] for k in ("resolution", "split", "sample_id")})).hexdigest()
    records = []
    for model in route["cost_models"]:
        for variant in route["task_variants"]:
            task = {"row_id": row_id, "cost_model": model, "task_variant": variant}
            records.append(
                {k: entry[k] for k in ("resolution", "split", "sample_id")}
                | {
                    "cost_model": model,
                    "task_variant": variant,
                    "legal": True,
                    "stored_cost_agreement": model == "energy",
                    "qualified": model == "energy",
                    "optimality": model == "energy",
                    "reference_status": "exact" if model == "energy" else "budget_exhausted",
                    "task_id": hashlib.sha256(canonical(task)).hexdigest(),
                }
            )
    finding = entry | {
        "source": source,
        "row_id": row_id,
        "source_stat": shard,
        "protocol_sha256": route_id,
        "roster_sha256": roster_id,
        "records": records,
    }
    finding_id = hashlib.sha256(canonical(finding)).hexdigest()
    finding["finding_sha256"] = finding_id
    finding_path = f"findings/{row_id}.json"
    write(root, f"route-audit/{finding_path}", finding)
    counts = {
        "rows": 1,
        "task_records": 6,
        "legal": 6,
        "stored_cost_agreement": 2,
        "reference_exact": 2,
        "reference_censored": 4,
        "reference_no_path": 0,
        "reference_not_run": 0,
        "qualified": 2,
    }
    summary = {
        "protocol_version": route["protocol_version"],
        "source": source,
        "protocol_sha256": route_id,
        "metadata_protocol_sha256": metadata_id,
        "roster_sha256": roster_id,
        "population_estimation": False,
        "status": "complete",
        "counts": counts,
        "by_resolution": {"64": counts},
        "coverage": {"selected_rows": 1, "completed_rows": 1, "expected_task_records": 6, "completed_task_records": 6},
        "findings": [{"row_id": row_id, "path": finding_path, "finding_sha256": finding_id}],
    }
    write(root, "route-audit/summary.json", summary)
    return root / f"route-audit/{finding_path}"


def test_stratified_counts_preserve_censoring_and_denominators(tmp_path):
    from passage.qualification_reports import qualification_report

    route_fixture(tmp_path)
    report = qualification_report(tmp_path)
    assert report["completion"]["route_audit"] is True
    assert report["completion"]["full_campaign"] is False
    route = report["resolutions"][0]["routes"]
    assert route["requested_variants"] == route["task_records"] == 6
    assert route["legal"] == 6
    assert route["stored_cost_agreement"] == route["qualified"] == 2
    assert route["reference_censored"] == 4
    assert report["counts"]["population_rows"] == 4


def test_finding_tamper_and_relaxed_tolerance_rejected(tmp_path):
    from passage.qualification_reports import qualification_report

    path = route_fixture(tmp_path)
    finding = json.loads(path.read_text())
    finding["records"][0]["qualified"] = True
    path.write_text(json.dumps(finding))
    with pytest.raises(ValueError, match="hash"):
        qualification_report(tmp_path)
    route_fixture(tmp_path)
    path = tmp_path / "route-audit/protocol.json"
    protocol = json.loads(path.read_text())
    protocol["rtol"] = 1e-2
    write(tmp_path, "route-audit/protocol.json", protocol)
    summary = json.loads((tmp_path / "route-audit/summary.json").read_text())
    summary["protocol_sha256"] = hashlib.sha256(canonical(protocol)).hexdigest()
    write(tmp_path, "route-audit/summary.json", summary)
    with pytest.raises(ValueError, match="budget"):
        qualification_report(tmp_path)


def test_cached_census_adapter_preserves_unknown_means(tmp_path):
    from passage.dataset import DatasetSource
    from passage.qualification_reports import dataset_census_report

    metadata_fixture(tmp_path)
    report = dataset_census_report(DatasetSource(None, repo=REPO, revision=PIN), 64, tmp_path)
    assert report["population"] == "full_core"
    assert report["total_rows"] == 4
    assert all(s["mean_relief_m"] is None and s["mean_obstacle_fraction"] is None for s in report["splits"])
    assert report["source"]["content_verified"] is False
    assert "fixed bins" in report["statistic_scope"]


def test_cached_census_rejects_changed_local_source(tmp_path):
    from passage.dataset import DatasetSource
    from passage.qualification import _source_stat
    from passage.qualification_reports import dataset_census_report

    root = tmp_path / "data"
    source = DatasetSource(root, repo=REPO, revision=PIN)
    evidence = tmp_path / "evidence"
    metadata_fixture(evidence)
    protocol = json.loads((evidence / "metadata/protocol.json").read_text())
    protocol["source"] = source.identity()
    for cell in protocol["manifest"].values():
        shard = cell["shards"][0]
        path = root / shard["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny fixture")
        shard.update(_source_stat(source, shard["path"]))
    summary = json.loads((evidence / "metadata/summary.json").read_text())
    summary.update(source=source.identity(), protocol_sha256=hashlib.sha256(canonical(protocol)).hexdigest())
    write(evidence, "metadata/protocol.json", protocol)
    write(evidence, "metadata/summary.json", summary)
    assert dataset_census_report(source, 64, evidence)["total_rows"] == 4
    (root / "64x64/train/part.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        dataset_census_report(source, 64, evidence)


def test_authentication_requires_log_objects_not_just_complete_receipt(tmp_path):
    from passage.qualification_reports import qualification_report

    manifest = {
        "schema_version": 1,
        "root": "/fixture",
        "repo": REPO,
        "revision": PIN,
        "repo_type": "dataset",
        "files": [{"path": "a", "size": 10, "algorithm": "sha256", "expected_hash": "b" * 64}],
    }
    manifest_id = hashlib.sha256(canonical(manifest) + b"\n").hexdigest()
    write(tmp_path, "source-authentication/manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "root": "/fixture",
        "repo": REPO,
        "revision": PIN,
        "manifest_sha256": manifest_id,
        "status": "complete",
        "content_verified": True,
        "required_files": 1,
        "required_bytes": 10,
        "verified_files": 1,
        "verified_bytes": 10,
        **dict.fromkeys(
            ("mismatched_files", "missing_files", "changed_files", "error_files", "interrupted_files", "pending_files"),
            0,
        ),
    }
    write(tmp_path, "source-authentication/receipt.json", receipt)
    assert qualification_report(tmp_path)["completion"]["source_authentication"] is False
    stat = {"device": 1, "inode": 2, "size": 10, "mtime_ns": 3, "ctime_ns": 4}
    log = {
        "path": "a",
        "status": "verified",
        "expected_hash": "b" * 64,
        "observed_hash": "b" * 64,
        "manifest_sha256": manifest_id,
        "stat": stat,
        "stat_after": stat,
    }
    write(tmp_path, "source-authentication/progress.jsonl", log)
    assert qualification_report(tmp_path)["completion"]["source_authentication"] is True


@requires_paper_checkout
def test_paper_exporter_missing_prerequisites(tmp_path):
    module = load_paper_figure_script("qualification_paper_test")
    receipt = module.render(tmp_path / "missing", tmp_path / "generated", tmp_path / "summary.tex")
    assert receipt["status"] == "not_run"
    tex = (tmp_path / "summary.tex").read_text()
    assert "planned" in tex and "pending" in tex and "not completed results" in tex
    assert (tmp_path / "generated/qualification_receipt.json").is_file()
    assert (tmp_path / "qualification_geography.tex").is_file()
    assert "pending" in (tmp_path / "qualification_geography.tex").read_text()
    assert "population defect rates" in (tmp_path / "qualification_routes.tex").read_text()
    assert (tmp_path / "generated/qualification_tables.json").is_file()
    with pytest.raises(ValueError, match="Completed metadata census"):
        module.render(tmp_path / "missing", tmp_path / "blocked", tmp_path / "blocked.tex", require_complete=True)
    assert not (tmp_path / "blocked").exists()


def test_portable_notebook_missing_prerequisites(tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "paper/notebooks/dataset_qualification.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        tags = set(cell.get("metadata", {}).get("tags", []))
        # The receipt checks read the published data; the bundle section needs only its own inputs.
        if cell["cell_type"] != "code" or not tags & {"parameters", "setup", "qualification-bundle"}:
            continue
        exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)
        if "parameters" in tags:
            namespace.update(
                project_path=str(notebook_path.parents[2]),
                output_dir=str(tmp_path / "notebook"),
                qualification_root=str(tmp_path / "missing"),
            )
    bundle = namespace["bundle"]
    assert bundle["status"] == "not_run"
    assert all(value == "not_run" for value in namespace["bundle_sections"].values())
    reported_hash = bundle["binding"].pop("report_sha256")
    assert reported_hash == hashlib.sha256(canonical(bundle)).hexdigest()
    assert {Path(p).suffix for p in bundle["figures"]} == {".pdf", ".png"}
    assert (tmp_path / "notebook/dataset_qualification-bundle.json").is_file()


def rectangle_denominator_fixture(root):
    route_fixture(root)
    summary = json.loads((root / "metadata/summary.json").read_text())
    for cell in summary["cells"].values():
        cell.update(
            footprints={"valid": 1},
            resolution_mismatch=0,
            split_mismatch=0,
            source_tiles={"N001E001": 1},
            support_tiles={"N001E001": 1, "N001E002": 1},
        )
    summary["cells"]["64/train"]["footprints"] = {"missing": 1}
    summary["geography"] = {
        "64": {
            "status": "complete",
            "source_cell_windows": "unknown",
            "independence_claim": False,
            "split_pairs": {
                "train/test": {
                    "shared_source_tiles": 1,
                    "shared_support_tiles": 2,
                    "positive_overlap_rows": {"train": 0, "test": 0},
                    "boundary_contact_rows": {"train": 0, "test": 0},
                }
            },
        }
    }
    write(root, "metadata/summary.json", summary)
    route = json.loads((root / "route-audit/protocol.json").read_text())
    route["metadata_summary_sha256"] = hashlib.sha256(canonical(summary)).hexdigest()
    write(root, "route-audit/protocol.json", route)
    # This fixture only needs metadata table evidence; route artifacts from an
    # earlier metadata binding are omitted rather than relabeled.
    (root / "route-audit/protocol.json").unlink()
    (root / "route-audit/summary.json").unlink()


def test_receipt_tables_keep_full_and_valid_rectangle_denominators(tmp_path):
    from passage.qualification_reports import qualification_report, qualification_table_data

    rectangle_denominator_fixture(tmp_path)
    tables = qualification_table_data(qualification_report(tmp_path))
    structural = tables["metadata"][0]
    assert structural["core_rows"] == 4
    assert structural["valid_rectangle_rows"] == 3
    assert structural["excluded_rectangle_rows"] == 1
    pair = tables["geography"][0]
    assert pair["left_full_rows"] == 1 and pair["left_valid_rectangle_rows"] == 0
    assert pair["right_full_rows"] == pair["right_valid_rectangle_rows"] == 1
    assert pair["shared_primary_tiles"] == 1 and pair["shared_support_tiles"] == 2
    assert tables["routes"][0]["planned_variants"] == 168
    assert tables["routes"][0]["requested_variants"] == 6
    assert tables["routes"][0]["completed_variants"] is None


@requires_paper_checkout
def test_paper_geography_table_keeps_full_and_valid_rectangle_denominators(tmp_path):
    from passage.qualification_reports import qualification_report

    rectangle_denominator_fixture(tmp_path)
    module = load_paper_figure_script("qualification_tables_test")
    tex = module.qualification_geography_tex(qualification_report(tmp_path))
    assert r"\resizebox" not in tex
    assert r"\shortstack[r]{Full\\rows}" in tex and r"\shortstack[r]{Valid\\rows}" in tex
    assert r"\shortstack[r]{Positive\\overlap}" in tex and r"\shortstack[r]{Boundary\\contact}" in tex
    assert r"\shortstack[r]{Excluded\\rows}" in tex
    assert "Both use the explicit valid-row denominator" in tex and "not additive" in tex
    assert "rows, not intersecting pairs" in tex and "independence remains unknown" in tex
    assert r"64 & Train & 1 & 0 & 0 & 0 & 1 \\" in tex
    assert r"64 & Test & 1 & 1 & 0 & 0 & 0 \\" in tex
