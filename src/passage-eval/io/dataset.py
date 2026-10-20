"""Load released PASSAGE scenes, preserving calibration, task and oracle labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

from passage.dataset import DatasetSource, SPLITS, decode_scene_inputs
from passage_eval.metrics.cost import canonical_cost_model
from passage_eval.metrics.reliability import path_status
from passage_eval.protocol.planner import Scene

PAPER_DATASET_REPO = "thalesgroup/passage"



def scene_from_row(
    row: Mapping[str, Any],
    *,
    resolution: int,
    cost_model: str,
    split: str,
    cost_weight: float | None = None,
    task_variant: str = "obstacles",
) -> Scene:
    """Decode one released row; reject malformed or mismatched task metadata.

    The weight comes from the sample's solver configuration. An override may
    confirm that value but cannot redefine the task while keeping its old oracle.
    """
    sample_id = str(row.get("id", "<missing-id>"))
    try:
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("missing or empty sample ID")
        split = SPLITS[split]
        model = canonical_cost_model(cost_model)
        if task_variant not in {"obstacles", "free"}:
            raise ValueError("task_variant must be obstacles or free")
        inputs = decode_scene_inputs(row, resolution=resolution, split=split)
        metadata = inputs["metadata"]
        metadata["task_variant"] = task_variant
        tensor = inputs["input_tensor"]
        elevation = inputs["elevation"]
        obstacles = inputs["obstacles"]
        start, goal = inputs["start"], inputs["goal"]
        if task_variant == "free":
            obstacles[:] = False
            tensor[:, :, 2] = 0
        matches = [
            name for name, cfg in metadata["solvers"].items() if canonical_cost_model(cfg["cost_model"]) == model
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one released solver for {model}")
        solver_name = matches[0]
        cfg = metadata["solvers"][solver_name]
        weight = float(cfg["weight"])
        if not np.isfinite(weight) or weight < 0 or cfg["diagonal_movement"] != "always":
            raise ValueError("invalid released weight or connectivity")
        if cost_weight is not None and not np.isclose(cost_weight, weight, rtol=1e-6, atol=0):
            raise ValueError("cost weight override differs from released oracle weight")
        oracle = row[f"path_{solver_name}_{task_variant}"]
        connected, violated, _ = path_status(oracle, start=start, goal=goal, obstacles=obstacles)
        if not connected or violated:
            raise ValueError("released oracle is not a valid path")
        oracle_cost = float(metadata["paths"][solver_name][task_variant]["cost"])
        if not np.isfinite(oracle_cost) or oracle_cost <= 0:
            raise ValueError("released oracle cost must be positive and finite")
        return Scene(
            sample_id,
            elevation,
            obstacles,
            start,
            goal,
            model,
            weight,
            [tuple(map(int, p)) for p in oracle],
            oracle_cost,
            metadata,
            tensor,
        )
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as exc:
        raise ValueError(f"PASSAGE sample {sample_id}: {exc}") from exc
    except zstandard.ZstdError as exc:
        raise ValueError(f"PASSAGE sample {sample_id}: corrupt tensor") from exc


def _local_rows(root: Path, resolution: int, split: str) -> Iterable[dict[str, Any]]:
    yield from DatasetSource(root).iter_rows(resolution, split)


def load_val_scenes(
    *,
    resolution: int,
    cost_model: str,
    n_samples: int = 1024,
    split: str = "validation",
    cost_weight: float | None = None,
    dataset_path: str | Path | None = None,
    dataset_repo: str = PAPER_DATASET_REPO,
    dataset_revision: str | None = None,
    task_variant: str = "obstacles",
) -> list[Scene]:
    """Load a deterministic prefix of a local export or a Hugging Face stream.

    Local roots use ``<N>x<N>/<split>/data-*.parquet``. Remote roots use
    the full ``<N>x<N>`` configuration, never the preview ``_samples`` config.
    Dataset errors propagate; no synthetic data or replacement oracle is created.
    """
    if split not in SPLITS or resolution <= 0 or n_samples <= 0:
        raise ValueError("Require a known split and positive resolution/sample count")
    split = SPLITS[split]
    cost_model = canonical_cost_model(cost_model)
    source = DatasetSource(dataset_path, repo=dataset_repo, revision=dataset_revision)
    # The selected route and metadata suffice; image/other-objective columns
    # are never read while preparing one benchmark cell.
    suffix = "slope" if cost_model == "slope_uphill" else cost_model
    columns = ["id", "tensor", "metadata", f"path_astar_{suffix}_{task_variant}"]
    rows = source.iter_rows(resolution, split, columns=None if dataset_path is not None else columns, limit=n_samples)
    scenes: list[Scene] = []
    seen: set[str] = set()
    # Do not read or decode another row after satisfying the requested cap.
    try:
        for _ in range(n_samples):
            row = next(rows, None)
            if row is None:
                break
            scene = scene_from_row(
                row,
                resolution=resolution,
                cost_model=cost_model,
                split=split,
                cost_weight=cost_weight,
                task_variant=task_variant,
            )
            if scene.sample_id in seen:
                raise ValueError(f"Duplicate PASSAGE sample {scene.sample_id}")
            seen.add(scene.sample_id)
            scenes.append(scene)
    finally:
        if hasattr(rows, "close"):
            rows.close()
    if not scenes:
        raise ValueError("The selected PASSAGE split is empty")
    return scenes
