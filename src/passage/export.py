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

"""Build PASSAGE release artifacts for compression, export, and upload stages."""

import csv
import hashlib
import io
import tempfile
from collections import Counter
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import httpcore
import httpx
import numpy as np
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Features, Image, List, Sequence, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger
from PIL import Image as PILImage

from passage.utils import load_tensor
from passage.visualization import tensor_to_image

try:  # Optional papermill fallback
    import papermill as pm

    PAPERMILL_AVAILABLE = True
except Exception:  # pragma: no cover - papermill optional
    PAPERMILL_AVAILABLE = False


CONFIG_NAME_PATTERN = re.compile(r"^(?P<resolution>\d+)x(?P=resolution)(?P<sample>_samples)?$")
DEFAULT_REPO_ID = "thalesgroup/passage"
DEFAULT_SPLIT_NAMES = ("train", "calibration", "validation", "test")
EXPORT_ROOT_METADATA_FILES = (
    "README.md",
    "croissant.json",
    "DATASHEET.md",
    "CITATION.cff",
    "LICENSE",
    "CHANGELOG.md",
    "DATASET_LICENSE.md",
)
GITATTRIBUTES_CONTENT = """\
# Hugging Face LFS — managed by `passage export`. Edit by regenerating with --force-gitattributes.
*.parquet filter=lfs diff=lfs merge=lfs -text
*.npy.zst filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
"""
DEFAULT_SAMPLE_SIZE_WARN_MB = 512
DEFAULT_PARQUET_SIZE_WARN_MB = 5_000  # HF hard cap is 50 GB; warn well before
DEFAULT_EXPORT_MEMORY_BUDGET_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class ExportSteps:
    """Per-step skip/force flags for prepare_export."""

    force_all: bool = False
    force_notebooks: bool = False
    force_assets: bool = False
    force_calibration: bool = False
    force_analysis_notebooks: bool = False
    force_metadata: bool = False
    force_readme: bool = False
    force_gitattributes: bool = False
    force_strip_outputs: bool = False
    skip_strip_outputs: bool = False
    skip_gitattributes: bool = False
    skip_validate_croissant: bool = False
    skip_validate_readme: bool = False
    skip_audit_compression: bool = False
    sample_size_warn_mb: int = DEFAULT_SAMPLE_SIZE_WARN_MB
    parquet_size_warn_mb: int = DEFAULT_PARQUET_SIZE_WARN_MB

    def force(self, attr: str) -> bool:
        return self.force_all or getattr(self, attr, False)


@dataclass(frozen=True)
class ReleaseContext:
    """Shared context for PASSAGE release stages."""

    config_file: Path
    dataset_path: Path
    calibration_file: Path
    calibrate_path: Path
    export_dirpath: Path
    chunk_sizes: dict[int, int]
    compression_level: int
    configured_default: str | None
    sample_size: int
    solver_names: list[str]
    notebooks_path: Path | None = None
    dataset_notebook: Path | None = None
    split_names: tuple[str, ...] = DEFAULT_SPLIT_NAMES


def parse_path_csv(csv_path: Path, *, expected_sha256: str | None = None) -> list[list[int]]:
    """Parse a path .csv file with rows "row,col" (no header) and return [[row, col], ...]."""

    out: list[list[int]] = []
    content = csv_path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Release admitted artifact content hash mismatch: {csv_path}")
    with io.StringIO(content.decode("utf-8"), newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"Malformed path CSV row in {csv_path}: expected exactly two coordinates")
            x = int(row[0].strip())
            y = int(row[1].strip())
            out.append([x, y])
    return out


def _load_export_metadata(content: str) -> dict:
    """Use the installed safe C parser, preserving SafeLoader types and fallback."""
    return yaml.load(content, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _save_png(image_path: Path, image_bgr) -> None:
    """Write an OpenCV-style BGR image as a PNG without requiring OpenCV."""
    rgb_image = np.ascontiguousarray(image_bgr[:, :, ::-1])
    PILImage.fromarray(rgb_image).save(image_path, format="PNG", compress_level=0)


def _ensure_preview_image_for_prepare(
    resolution: int,
    metadata: dict,
    tensor_filepath: Path,
    solver_name: str,
    path_free: list[list[int]],
    path_obstacles: list[list[int]],
    solver_image_filepath: Path,
    tensor=None,
):
    """Generate a missing solver preview image from saved dataset artifacts."""
    if solver_image_filepath.exists():
        return tensor

    if not tensor_filepath.exists():
        raise FileNotFoundError(
            f"Cannot generate missing preview image because tensor file is missing: {tensor_filepath}"
        )

    if tensor is None:
        tensor = load_tensor(tensor_filepath, (resolution, resolution, 3))

    obstacles_enabled = bool((metadata or {}).get("obstacles", {}).get("enabled", False))
    alt_path = path_obstacles if obstacles_enabled and path_obstacles != path_free else None
    preview_image = tensor_to_image(tensor, path_pixels=path_free, alt_path_pixels=alt_path)

    solver_image_filepath.parent.mkdir(parents=True, exist_ok=True)
    _save_png(solver_image_filepath, preview_image)
    logger.info(f"Generated missing preview image for export ({solver_name}): {solver_image_filepath}")
    return tensor


def _process_meta_for_prepare(
    resolution: int,
    sample: bool,
    solver_names: list[str],
    metadata_filepath: str,
    *,
    preview_root: Path | None = None,
    verified_receipt_sha256: str | None = None,
) -> dict:
    """Process a single metadata file into a serializable dict for HF dataset rows."""

    metadata_filepath = Path(metadata_filepath)
    split_dirpath = Path(metadata_filepath.parent.parent.parent.parent)
    lat_dir = metadata_filepath.parent.parent.name
    lon_dir = metadata_filepath.parent.name
    stem = metadata_filepath.stem

    tensor_filepath = split_dirpath / "tensors" / lat_dir / lon_dir / f"{stem}.npy.zst"
    if not tensor_filepath.exists():
        raise FileNotFoundError(f"Missing required tensor for export: {tensor_filepath}")

    metadata_content = metadata_filepath.read_bytes()
    metadata = _load_export_metadata(metadata_content.decode("utf-8")) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Export metadata must be a mapping")
    expected_identity = {"index": int(stem), "resolution": resolution, "split": split_dirpath.name}
    if any(key in metadata and metadata[key] != value for key, value in expected_identity.items()):
        raise ValueError(f"Export scene identity mismatch: {metadata_filepath}")
    receipt = None
    if metadata.get("dataset_version"):
        from passage.successor_generation import validate_scene_receipt

        if metadata.get("units") != "metres":
            raise ValueError("Release export requires canonical metres units")
        if verified_receipt_sha256 is None:
            receipt = validate_scene_receipt(split_dirpath, int(stem))
        else:
            # Admission already validates all receipt contracts in a worker. Bind
            # this load to that exact receipt, and hash the actual bytes used to
            # build the row so a concurrent raw change cannot cross that boundary.
            receipt_content = (split_dirpath / "receipts" / f"{stem}.json").read_bytes()
            if hashlib.sha256(receipt_content).hexdigest() != verified_receipt_sha256:
                raise ValueError("Release admitted receipt content hash mismatch")
            receipt = json.loads(receipt_content)
            if hashlib.sha256(metadata_content).hexdigest() != receipt["artifacts"]["metadata"]["sha256"]:
                raise ValueError("Release admitted metadata content hash mismatch")
        if receipt["scene_status"] != "complete":
            raise ValueError("Only complete scene receipts can supply tensor rows")
    elif verified_receipt_sha256 is not None:
        raise ValueError("Release admitted metadata lacks dataset identity")
    tensor = None

    tensor_content = tensor_filepath.read_bytes()
    if (
        verified_receipt_sha256 is not None
        and hashlib.sha256(tensor_content).hexdigest() != receipt["artifacts"]["tensor"]["sha256"]
    ):
        raise ValueError("Release admitted tensor content hash mismatch")

    sample_dict = {
        "id": str(stem),
        "tensor": tensor_content,
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
        "scene_receipt": json.dumps(receipt, sort_keys=True) if receipt is not None else None,
    }
    if sample and verified_receipt_sha256 is not None:
        from passage.dataset import decode_scene_inputs

        tensor = decode_scene_inputs(sample_dict, resolution=resolution, split=split_dirpath.name)["input_tensor"]

    # Load paths dynamically for each solver and per-solver images (if sample)
    for solver_name in solver_names:
        path_free_filepath = split_dirpath / "paths" / solver_name / "free" / lat_dir / lon_dir / f"{stem}.csv"
        path_obstacles_filepath = (
            split_dirpath / "paths" / solver_name / "obstacles" / lat_dir / lon_dir / f"{stem}.csv"
        )
        for path_filepath in (path_free_filepath, path_obstacles_filepath):
            if not path_filepath.is_file():
                raise FileNotFoundError(f"Missing required path CSV for export: {path_filepath}")
        hashes = receipt["artifacts"] if verified_receipt_sha256 is not None else {}
        path_free = parse_path_csv(
            path_free_filepath, expected_sha256=hashes.get(f"path:{solver_name}:free", {}).get("sha256")
        )
        path_obstacles = parse_path_csv(
            path_obstacles_filepath, expected_sha256=hashes.get(f"path:{solver_name}:obstacles", {}).get("sha256")
        )
        for variant, path in (("free", path_free), ("obstacles", path_obstacles)):
            stats = metadata.get("paths", {}).get(solver_name, {}).get(variant, {})
            if not path and stats.get("status") not in ("infeasible", "failed"):
                raise ValueError("An empty CSV requires an explicit terminal status")
            if stats.get("status") in ("infeasible", "failed") and (path or stats.get("cost") is not None):
                raise ValueError("Terminal route status requires empty path and null cost")

        # Per-solver image file (sample-only)
        solver_image_filepath = (
            (preview_root or split_dirpath / "images") / solver_name / lat_dir / lon_dir / f"{stem}.png"
        )
        if sample:
            if not solver_image_filepath.exists():
                tensor = _ensure_preview_image_for_prepare(
                    resolution=resolution,
                    metadata=metadata,
                    tensor_filepath=tensor_filepath,
                    solver_name=solver_name,
                    path_free=path_free,
                    path_obstacles=path_obstacles,
                    solver_image_filepath=solver_image_filepath,
                    tensor=tensor,
                )
            sample_dict[f"image_{solver_name}"] = {"bytes": solver_image_filepath.read_bytes()}

        sample_dict[f"path_{solver_name}_free"] = path_free
        sample_dict[f"path_{solver_name}_obstacles"] = path_obstacles
    if receipt is not None:
        _validate_canonical_export_row(sample_dict, resolution, split_dirpath.name, solver_names)
    return sample_dict


def _validate_canonical_export_row(row: dict, resolution: int, split: str, solver_names: list[str]) -> None:
    """Reconstruct authoritative terrain and recost independently of receipt hashes."""
    from passage.dataset import decode_scene_inputs
    from passage_eval.metrics.cost import path_cost
    from passage_eval.metrics.reliability import path_status
    from passage_eval.protocol.contracts import GraphSpec

    inputs = decode_scene_inputs(row, resolution=resolution, split=split)
    metadata = row["metadata"]
    if set(metadata["solvers"]) != set(solver_names) or set(metadata["paths"]) != set(solver_names):
        raise ValueError("Release export solver identity mismatch")
    for solver in solver_names:
        config = metadata["solvers"][solver]
        graph = GraphSpec(**config["graph_spec"])
        from passage_eval.metrics.cost import canonical_cost_model

        if graph.cost_model != canonical_cost_model(config["cost_model"]) or graph.cost_weight != config["weight"]:
            raise ValueError("Release export graph/objective mismatch")
        for variant in ("free", "obstacles"):
            stats = metadata["paths"][solver][variant]
            path = row[f"path_{solver}_{variant}"]
            if stats["length"] != len(path):
                raise ValueError("Release export route length mismatch")
            status = stats.get("status")
            if status in ("infeasible", "failed"):
                if path or stats.get("cost") is not None:
                    raise ValueError("Terminal route status requires empty path and null cost")
                continue
            if status != "ok" or not path:
                raise ValueError("Release export requires explicit valid route status")
            obstacles = inputs["obstacles"] if variant == "obstacles" else np.zeros_like(inputs["obstacles"])
            connected, violation, _ = path_status(path, start=inputs["start"], goal=inputs["goal"], obstacles=obstacles)
            actual = path_cost(inputs["elevation"], path, weight=graph.cost_weight, cost_model=graph.cost_model)
            if not connected or violation or not np.isfinite(actual):
                raise ValueError("Release export route is illegal on decoded terrain")
            stored = stats.get("cost")
            if stored is None or not np.isclose(actual, stored, rtol=graph.rtol, atol=graph.atol):
                raise ValueError("Release export canonical route cost mismatch")


def _solver_feature() -> dict:
    """Return the feature schema for a single solver config."""
    return {
        "backend": Value("string"),
        "name": Value("string"),
        "diagonal_movement": Value("string"),
        "weight": Value("float64"),
        "heuristic": Value("string"),
        "cost_model": Value("string"),
        "solver": Value("string"),
        "max_marker_retries": Value("int32"),
        "graph_spec": {
            "connectivity": Value("int32"),
            "diagonal_policy": Value("string"),
            "pixel_size_m": Value("float64"),
            "obstacle_policy": Value("string"),
            "coordinate_order": Value("string"),
            "cost_model": Value("string"),
            "cost_weight": Value("float64"),
            "rtol": Value("float64"),
            "atol": Value("float64"),
        },
    }


def _path_stats_feature() -> dict:
    """Return the feature schema for path stats (free or obstacles)."""
    return {
        "length": Value("int32"),
        "crosses_obstacles": Value("bool"),
        "considered_area": Value("int32"),
        "cost": Value("float64"),
        "status": Value("string"),
    }


def _path_stats_obstacles_feature() -> dict:
    """Return the feature schema for path stats with obstacles."""
    return {
        "length": Value("int32"),
        "crosses_obstacles": Value("bool"),
        "considered_area": Value("int32"),
        "cost": Value("float64"),
        "status": Value("string"),
        "cost_diff_percent": Value("float32"),
    }


def _features(sample: bool, solver_names: list[str]) -> Features:
    features_dict = {
        "id": Value("string"),
        "metadata_json": Value("string"),
        "scene_receipt": Value("string"),
    }

    # Add per-solver image columns (sample-only) and dynamic path columns for each solver
    for solver_name in solver_names:
        if sample:
            features_dict[f"image_{solver_name}"] = Image()
        features_dict[f"path_{solver_name}_free"] = Sequence(Sequence(Value("int32")))
        features_dict[f"path_{solver_name}_obstacles"] = Sequence(Sequence(Value("int32")))

    # Build solvers feature as a dict with solver names as keys
    solvers_feature = {name: _solver_feature() for name in solver_names}

    # Build paths feature as a dict with solver names as keys
    paths_feature = {
        name: {
            "free": _path_stats_feature(),
            "obstacles": _path_stats_obstacles_feature(),
            "timing": {
                "path_free_seconds": Value("float32"),
                "path_obstacles_seconds": Value("float32"),
                "cost_compute_seconds": Value("float32"),
                "total_seconds": Value("float32"),
            },
        }
        for name in solver_names
    }

    features_dict["metadata"] = {
        "dataset_version": Value("string"),
        "units": Value("string"),
        "index": Value("int32"),
        "resolution": Value("int32"),
        "split": Value("string"),
        "calibration": {
            "min": Value("float64"),
            "max": Value("float64"),
        },
        "tile": {
            "source": {
                "lat": Value("int32"),
                "lon": Value("int32"),
                "name": Value("string"),
            },
            "stats": {
                "min_elevation": Value("float32"),
                "max_elevation": Value("float32"),
            },
            "used": List(
                {
                    "lat": Value("int32"),
                    "lon": Value("int32"),
                    "name": Value("string"),
                }
            ),
            "overlapping": List(
                {
                    "lat": Value("int32"),
                    "lon": Value("int32"),
                    "name": Value("string"),
                }
            ),
        },
        "crop": {
            "min_elevation": Value("float32"),
            "max_elevation": Value("float32"),
            "target_elevation": Value("float32"),
            "center": {
                "lat": Value("float32"),
                "lon": Value("float32"),
            },
            "top_left": {
                "lat": Value("float32"),
                "lon": Value("float32"),
            },
            "bottom_right": {
                "lat": Value("float32"),
                "lon": Value("float32"),
            },
        },
        "markers": {
            "start_point": {
                "i": Value("int32"),
                "j": Value("int32"),
                "elevation": Value("float32"),
            },
            "end_point": {
                "i": Value("int32"),
                "j": Value("int32"),
                "elevation": Value("float32"),
            },
        },
        "obstacles": {
            "enabled": Value("bool"),
            "ratio_max": Value("float32"),
            "ratio_target": Value("float32"),
            "ratio_actual": Value("float32"),
            "size_min_ratio": Value("float32"),
            "size_max_ratio": Value("float32"),
            "penalty": Value("float32"),
        },
        "provenance": {
            "generation_seed": Value("int64"),
            "sample_seed": Value("int64"),
            "calibration_sha256": Value("string"),
            "provenance_id": Value("string"),
        },
        "solvers": solvers_feature,
        "paths": paths_feature,
        "timing": {
            "tile_loading_seconds": Value("float32"),
            "marker_search_seconds": Value("float32"),
            "obstacle_generation_seconds": Value("float32"),
            "tensor_save_seconds": Value("float32"),
            "metadata_write_seconds": Value("float32"),
            "image_conversion_seconds": Value("float32"),
            "image_write_seconds": Value("float32"),
        },
        "timestamp": Value("timestamp[us]"),
    }

    features_dict["tensor"] = Value("binary")
    return Features(features_dict)


def _execute_notebook_with_papermill(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
    cell_timeout: int | None = None,
    job_timeout: int | None = None,
) -> bool:
    """Execute a notebook with papermill using uv, with fallback to papermill API."""

    if cell_timeout is not None or job_timeout is not None:
        from passage.notebook_export import execute_notebook

        try:
            execute_notebook(
                input_notebook,
                output_notebook,
                parameters,
                cwd=cwd or Path.cwd(),
                cell_timeout=cell_timeout or 300,
                job_timeout=job_timeout or 1800,
                offline=parameters.get("dataset_path") is not None,
            )
            return True
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.error(f"Notebook execution failed: {exc}")
            return False

    cmd_base = [
        "uv",
        "run",
        "papermill",
        str(input_notebook.resolve()),
        str(output_notebook.resolve()),
        "-k",
        "python3",
    ]
    for k, v in (parameters or {}).items():
        cmd_base.extend(["-p", str(k), str(v)])

    logger.info(f"Running notebook with: {' '.join(cmd_base)}")
    try:
        subprocess.run(cmd_base, cwd=str(cwd) if cwd else None, check=True)
        logger.info(f"✓ Notebook executed (via uv run): {output_notebook.name}")
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(f"uv run papermill failed (returncode={exc.returncode}); attempting papermill fallback")
        if PAPERMILL_AVAILABLE:
            try:
                pm.execute_notebook(
                    str(input_notebook.resolve()),
                    str(output_notebook.resolve()),
                    parameters=parameters or {},
                    kernel_name="python3",
                )
                logger.info(f"✓ Notebook executed via papermill fallback: {output_notebook.name}")
                return True
            except Exception as pm_exc:  # pragma: no cover - depends on papermill runtime
                logger.error(f"Papermill fallback failed: {pm_exc}")
                return False
        logger.warning("Papermill fallback unavailable; skipping notebook execution")
        return False


def _execute_notebook_with_retries(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
    max_retries: int | None = 3,
    retry_delay: int = 5,
    cell_timeout: int | None = None,
    job_timeout: int | None = None,
) -> bool:
    attempt = 0
    notebook_name = input_notebook.name
    while True:
        attempt += 1
        resolution_info = f" (resolution={parameters.get('RESOLUTION')})" if "RESOLUTION" in parameters else ""
        logger.info(f"Executing notebook {notebook_name} (attempt {attempt}){resolution_info}")
        try:
            if cell_timeout is not None or job_timeout is not None:
                success = _execute_notebook_with_papermill(
                    input_notebook, output_notebook, parameters, cwd, cell_timeout=cell_timeout, job_timeout=job_timeout
                )
            else:
                success = _execute_notebook_with_papermill(input_notebook, output_notebook, parameters, cwd)
        except Exception as exc:  # pragma: no cover - papermill errors vary
            logger.warning(f"Notebook {notebook_name} failed (attempt {attempt}): {exc}")
            success = False

        if success:
            return True

        if max_retries is None:
            time.sleep(max(0, int(retry_delay)))
            continue

        if attempt >= max_retries:
            return False

        time.sleep(max(0, int(retry_delay)))


def _dir_size_bytes(path: Path, sample: bool) -> int:
    if not path.exists():
        return 0
    if sample:
        return int(sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and f.suffix != ".png"))
    else:
        return int(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))


def _dir_num_files(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(1 for f in path.rglob("*") if f.is_file()))


def _config_name(resolution: int, sample: bool) -> str:
    return f"{resolution}x{resolution}" + ("_samples" if sample else "")


def _parse_config_name(config_name: str) -> tuple[int, bool] | None:
    match = CONFIG_NAME_PATTERN.fullmatch(config_name)
    if match is None:
        return None
    return int(match.group("resolution")), bool(match.group("sample"))


def _sort_config_names(config_names: set[str] | list[str]) -> list[str]:
    parsed_config_names: list[tuple[int, int, str]] = []
    for config_name in set(config_names):
        parsed = _parse_config_name(config_name)
        if parsed is None:
            continue
        resolution, sample = parsed
        parsed_config_names.append((resolution, 0 if sample else 1, config_name))
    return [config_name for _, _, config_name in sorted(parsed_config_names)]


def _infer_remote_config_names(repo_files: list[str]) -> set[str]:
    config_names: set[str] = set()
    for repo_file in repo_files:
        top_level = repo_file.split("/", 1)[0]
        if _parse_config_name(top_level) is not None:
            config_names.add(top_level)
    return config_names


def _select_default_config(config_names: set[str] | list[str], configured_default: str | None) -> str | None:
    sorted_config_names = _sort_config_names(config_names)
    config_name_set = set(sorted_config_names)

    if configured_default and configured_default in config_name_set:
        return configured_default

    sample_configs = [config_name for config_name in sorted_config_names if config_name.endswith("_samples")]
    if sample_configs:
        return sample_configs[0]

    full_configs = [config_name for config_name in sorted_config_names if not config_name.endswith("_samples")]
    if full_configs:
        return full_configs[0]

    return None


def _cleanup_incremental_export_dirs(export_dirpath: Path, resolutions: list[int]) -> None:
    for resolution in sorted(set(resolutions)):
        for sample in [True, False]:
            config_dirpath = export_dirpath / _config_name(resolution, sample)
            if config_dirpath.exists():
                shutil.rmtree(config_dirpath)


def _build_incremental_upload_patterns(config_names: set[str] | list[str]) -> list[str]:
    allow_patterns = [*EXPORT_ROOT_METADATA_FILES, "assets/**", "calibration/**", "notebooks/**"]
    for config_name in _sort_config_names(config_names):
        allow_patterns.append(f"{config_name}/**")
    return allow_patterns


def _solver_feature_yaml() -> list[dict]:
    """Return the YAML feature schema for a single solver config."""
    return [
        {"name": "backend", "dtype": "string"},
        {"name": "name", "dtype": "string"},
        {"name": "diagonal_movement", "dtype": "string"},
        {"name": "weight", "dtype": "float32"},
        {"name": "heuristic", "dtype": "string"},
        {"name": "cost_model", "dtype": "string"},
        {"name": "solver", "dtype": "string"},
        {"name": "max_marker_retries", "dtype": "int32"},
    ]


def _path_stats_yaml() -> list[dict]:
    """Return the YAML feature schema for path stats."""
    return [
        {"name": "length", "dtype": "int32"},
        {"name": "crosses_obstacles", "dtype": "bool"},
        {"name": "considered_area", "dtype": "int32"},
        {"name": "cost", "dtype": "float32"},
    ]


def _path_stats_obstacles_yaml() -> list[dict]:
    """Return the YAML feature schema for path stats with obstacles."""
    return [
        {"name": "length", "dtype": "int32"},
        {"name": "crosses_obstacles", "dtype": "bool"},
        {"name": "considered_area", "dtype": "int32"},
        {"name": "cost", "dtype": "float32"},
        {"name": "cost_diff_percent", "dtype": "float32"},
    ]


def _timing_per_solver_yaml() -> list[dict]:
    """Return the YAML feature schema for per-solver timing."""
    return [
        {"name": "path_free_seconds", "dtype": "float32"},
        {"name": "path_obstacles_seconds", "dtype": "float32"},
        {"name": "cost_compute_seconds", "dtype": "float32"},
        {"name": "total_seconds", "dtype": "float32"},
    ]


def _generate_huggingface_yaml_header(
    repo_id: str,
    config_names: list[str] | set[str],
    split_names: list[str],
    default_config_name: str | None,
    solver_names: list[str],
) -> str:
    dataset_config = {
        "dataset_info": [],
        "configs": [],
        "language": ["en"],
        "license": ["other"],
        "task_categories": ["image-segmentation", "other"],
        "tags": ["elevation", "pathfinding", "dsm", "alos", "passage", "terrain", "obstacles", "A*"],
        "pretty_name": "PASSAGE",
        "size_categories": ["1M<n<10M"],
    }

    for config_name in _sort_config_names(config_names):
        parsed = _parse_config_name(config_name)
        if parsed is None:
            continue
        _, sample = parsed

        path_features = []
        for solver_name in solver_names:
            if sample:
                path_features.append({"name": f"image_{solver_name}", "dtype": "image"})
            path_features.append({"name": f"path_{solver_name}_free", "list": {"list": "int32"}})
            path_features.append({"name": f"path_{solver_name}_obstacles", "list": {"list": "int32"}})

        solvers_struct = [{"name": name, "struct": _solver_feature_yaml()} for name in solver_names]
        paths_struct = [
            {
                "name": name,
                "struct": [
                    {"name": "free", "struct": _path_stats_yaml()},
                    {"name": "obstacles", "struct": _path_stats_obstacles_yaml()},
                    {"name": "timing", "struct": _timing_per_solver_yaml()},
                ],
            }
            for name in solver_names
        ]

        dataset_info_config = {
            "config_name": config_name,
            "features": [
                {"name": "id", "dtype": "string"},
                {"name": "tensor", "dtype": "binary"},
                *path_features,
                {
                    "name": "metadata",
                    "struct": [
                        {"name": "index", "dtype": "int32"},
                        {"name": "resolution", "dtype": "int32"},
                        {"name": "split", "dtype": "string"},
                        {
                            "name": "calibration",
                            "struct": [
                                {"name": "min", "dtype": "float32"},
                                {"name": "max", "dtype": "float32"},
                            ],
                        },
                        {
                            "name": "tile",
                            "struct": [
                                {
                                    "name": "source",
                                    "struct": [
                                        {"name": "lat", "dtype": "int32"},
                                        {"name": "lon", "dtype": "int32"},
                                        {"name": "name", "dtype": "string"},
                                    ],
                                },
                                {
                                    "name": "stats",
                                    "struct": [
                                        {"name": "min_elevation", "dtype": "float32"},
                                        {"name": "max_elevation", "dtype": "float32"},
                                    ],
                                },
                                {
                                    "name": "used",
                                    "list": {
                                        "struct": [
                                            {"name": "lat", "dtype": "int32"},
                                            {"name": "lon", "dtype": "int32"},
                                            {"name": "name", "dtype": "string"},
                                        ]
                                    },
                                },
                                {
                                    "name": "overlapping",
                                    "list": {
                                        "struct": [
                                            {"name": "lat", "dtype": "int32"},
                                            {"name": "lon", "dtype": "int32"},
                                            {"name": "name", "dtype": "string"},
                                        ]
                                    },
                                },
                            ],
                        },
                        {
                            "name": "crop",
                            "struct": [
                                {"name": "min_elevation", "dtype": "float32"},
                                {"name": "max_elevation", "dtype": "float32"},
                                {"name": "target_elevation", "dtype": "float32"},
                                {
                                    "name": "center",
                                    "struct": [
                                        {"name": "lat", "dtype": "float32"},
                                        {"name": "lon", "dtype": "float32"},
                                    ],
                                },
                                {
                                    "name": "top_left",
                                    "struct": [
                                        {"name": "lat", "dtype": "float32"},
                                        {"name": "lon", "dtype": "float32"},
                                    ],
                                },
                                {
                                    "name": "bottom_right",
                                    "struct": [
                                        {"name": "lat", "dtype": "float32"},
                                        {"name": "lon", "dtype": "float32"},
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "markers",
                            "struct": [
                                {
                                    "name": "start_point",
                                    "struct": [
                                        {"name": "i", "dtype": "int32"},
                                        {"name": "j", "dtype": "int32"},
                                        {"name": "elevation", "dtype": "float32"},
                                    ],
                                },
                                {
                                    "name": "end_point",
                                    "struct": [
                                        {"name": "i", "dtype": "int32"},
                                        {"name": "j", "dtype": "int32"},
                                        {"name": "elevation", "dtype": "float32"},
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "obstacles",
                            "struct": [
                                {"name": "enabled", "dtype": "bool"},
                                {"name": "ratio_max", "dtype": "float32"},
                                {"name": "ratio_target", "dtype": "float32"},
                                {"name": "ratio_actual", "dtype": "float32"},
                                {"name": "size_min_ratio", "dtype": "float32"},
                                {"name": "size_max_ratio", "dtype": "float32"},
                                {"name": "penalty", "dtype": "float32"},
                            ],
                        },
                        {
                            "name": "provenance",
                            "struct": [
                                {"name": "generation_seed", "dtype": "int64"},
                                {"name": "sample_seed", "dtype": "int64"},
                            ],
                        },
                        {"name": "solvers", "struct": solvers_struct},
                        {"name": "paths", "struct": paths_struct},
                        {
                            "name": "timing",
                            "struct": [
                                {"name": "tile_loading_seconds", "dtype": "float32"},
                                {"name": "marker_search_seconds", "dtype": "float32"},
                                {"name": "obstacle_generation_seconds", "dtype": "float32"},
                                {"name": "tensor_save_seconds", "dtype": "float32"},
                                {"name": "metadata_write_seconds", "dtype": "float32"},
                                {"name": "image_conversion_seconds", "dtype": "float32"},
                                {"name": "image_write_seconds", "dtype": "float32"},
                            ],
                        },
                        {"name": "timestamp", "dtype": "timestamp[us]"},
                    ],
                },
            ],
        }
        configs_config = {
            "config_name": config_name,
            "data_files": [],
        }
        if config_name == default_config_name:
            configs_config["default"] = True
        for split in split_names:
            configs_config["data_files"].append(
                {
                    "split": split,
                    "path": f"{config_name}/{split}/*",
                }
            )
        dataset_info_config["features"] = _features(sample, solver_names)._to_yaml_list()
        dataset_config["dataset_info"].append(dataset_info_config)
        dataset_config["configs"].append(configs_config)
    # configs_yaml: list[str] = []
    # for res in sorted(resolutions):
    #     configs_yaml.append(f'  - config_name: "{res}x{res}"')
    #     configs_yaml.append("    data_files:")
    #     for split in sorted(split_names):
    #         configs_yaml.append(f"      - split: {split}")
    #         configs_yaml.append(f'        path: "{res}x{res}/{split}/data-*.parquet"')

    # dataset_infos: list[str] = []
    # for res in sorted(resolutions):
    #     dataset_infos.append(f" - config_name: {res}x{res}")
    #     dataset_infos.append("    features:")
    #     dataset_infos.append("      - name: id")
    #     dataset_infos.append("        dtype: string")
    #     dataset_infos.append("      - name: image")
    #     dataset_infos.append("        dtype: image")
    #     dataset_infos.append("      - name: tensor")
    #     dataset_infos.append("        list:")
    #     dataset_infos.append("          list:")
    #     dataset_infos.append("            list: float32")
    #     dataset_infos.append("      - name: path_free")
    #     dataset_infos.append("        list:")
    #     dataset_infos.append("          list: int32")
    #     dataset_infos.append("      - name: path_obstacles")
    #     dataset_infos.append("        list:")
    #     dataset_infos.append("          list: int32")
    #     dataset_infos.append("      - name: metadata")
    #     dataset_infos.append("        dtype: string")

    # header_lines = [
    #     "---",
    #     "license: mit",
    #     "task_categories:",
    #     "  - image-segmentation",
    #     "  - other",
    #     "tags:",
    #     "  - elevation",
    #     "  - pathfinding",
    #     "  - dsm",
    #     "  - alos",
    #     "  - passage",
    #     "  - terrain",
    #     # "dataset_info:",
    #     # "  features:",
    #     # "    - name: id",
    #     # "      dtype: string",
    #     # "    - name: image",
    #     # "      dtype: image",
    #     # "    - name: tensor",
    #     # "      dtype: binary",
    #     # "    - name: path_free",
    #     # "      sequence:",
    #     # "        sequence:",
    #     # "          dtype: int32",
    #     # "    - name: path_obstacles",
    #     # "      sequence:",
    #     # "        sequence:",
    #     # "          dtype: int32",
    #     # "    - name: metadata",
    #     # "      dtype: string",
    #     "configs:",
    #     *configs_yaml,
    #     "dataset_info:",
    #     '  description: "**PA**th **S**olving Data**S**et **A**nd **GE**nerator"',
    #     "  features:",
    #     "  - name: id",
    #     "    dtype: string",
    #     "  - name: image",
    #     "    dtype: image",
    #     "  - name: tensor",
    #     "    list:",
    #     "      list:",
    #     "        list: float32",
    #     "  - name: path_free",
    #     "    list:",
    #     "      list: int32",
    #     "  - name: path_obstacles",
    #     "    list:",
    #     "      list: int32",
    #     "  - name: metadata",
    #     "    dtype: string",
    #     "  configs:",
    #     *dataset_info_configs,
    #     "pretty_name: PASSAGE",
    #     "size_categories:",
    #     "  - 100K<n<1M",
    #     "---",
    #     "",
    # ]

    # return "\n".join(header_lines)
    return "---\n" + yaml.dump(dataset_config) + "\n---\n"


def _dataset_repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _dataset_terms_url(repo_id: str) -> str:
    return f"{_dataset_repo_url(repo_id)}/blob/main/DATASET_LICENSE.md"


def _customize_croissant(croissant_text: str, repo_id: str) -> str:
    croissant = json.loads(croissant_text)
    dataset_url = _dataset_repo_url(repo_id)
    croissant["@id"] = dataset_url
    croissant["url"] = dataset_url
    croissant["license"] = _dataset_terms_url(repo_id)

    for distribution in croissant.get("distribution", []):
        if distribution.get("@id") == "hf-passage-tree":
            distribution["contentUrl"] = f"{dataset_url}/tree/main"

    return json.dumps(croissant, indent=2, ensure_ascii=False) + "\n"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_export_dirpath(
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    export_dirpath: str | Path | None = None,
) -> Path:
    if export_dirpath is not None:
        return Path(export_dirpath).resolve()

    calibration_file = (
        Path(calibration_filepath).resolve()
        if calibration_filepath is not None
        else (Path(calibrate_dirpath).resolve() / "calibrate.json")
    )
    return calibration_file.parent.parent / "export"


def _resolve_config_file(config_filepath: str | Path, reference_dir: Path | None = None) -> Path:
    config_file = Path(config_filepath)
    if config_file.is_absolute():
        return config_file.resolve()

    cwd_candidate = config_file.resolve()
    if reference_dir is None:
        reference_dir = _repo_root()
    repo_relative_candidate = (reference_dir / config_file).resolve()
    if repo_relative_candidate.exists():
        return repo_relative_candidate
    return cwd_candidate


def _build_release_context(
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    notebooks_dirpath: str | Path | None = None,
    config_filepath: str | Path = "config/config.yaml",
    export_dirpath: str | Path | None = None,
    require_calibration: bool = True,
    require_notebooks: bool = False,
) -> ReleaseContext:
    dataset_path = Path(dataset_dirpath).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    calibration_file = (
        Path(calibration_filepath).resolve()
        if calibration_filepath is not None
        else (Path(calibrate_dirpath).resolve() / "calibrate.json")
    )
    if require_calibration and not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    notebooks_path = None
    dataset_notebook = None
    reference_dir = _repo_root()
    if notebooks_dirpath is not None:
        notebooks_path = Path(notebooks_dirpath).resolve()
        reference_dir = notebooks_path.parent
        if require_notebooks and not notebooks_path.exists():
            raise FileNotFoundError(f"Notebooks directory not found: {notebooks_path}")
        dataset_notebook = notebooks_path / "_dataset.ipynb"
        if require_notebooks and not dataset_notebook.exists():
            raise FileNotFoundError(f"Dataset notebook not found: {dataset_notebook}")

    config_file = _resolve_config_file(config_filepath, reference_dir)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    cfg = yaml.safe_load(config_file.read_text()) or {}
    export_cfg = cfg.get("export") or {}
    chunk_sizes = export_cfg.get("chunk_size")
    if not isinstance(chunk_sizes, dict) or not chunk_sizes:
        raise ValueError("Missing export.chunk_size in config.yaml")

    solver_names = [solver["name"] for solver in cfg.get("solvers", []) if "name" in solver]
    if not solver_names:
        raise ValueError("No solvers defined in config.yaml")

    sample_cfg = export_cfg.get("sample") or {}

    return ReleaseContext(
        config_file=config_file,
        dataset_path=dataset_path,
        calibration_file=calibration_file,
        calibrate_path=calibration_file.parent,
        export_dirpath=_resolve_export_dirpath(
            calibrate_dirpath=calibrate_dirpath,
            calibration_filepath=calibration_filepath,
            export_dirpath=export_dirpath,
        ),
        chunk_sizes=chunk_sizes,
        compression_level=int(export_cfg.get("compression_level", 12)),
        configured_default=export_cfg.get("default"),
        sample_size=int(sample_cfg.get("num_samples", 64)),
        solver_names=solver_names,
        notebooks_path=notebooks_path,
        dataset_notebook=dataset_notebook,
    )


def _normalize_cpus(num_cpus: int | None) -> int:
    return num_cpus if num_cpus is not None else (os.cpu_count() or 1)


def _resolve_dataset_resolutions(dataset_path: Path, resolutions: list[int] | None) -> list[int]:
    if resolutions is None:
        discovered = sorted(int(d.name) for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit())
        if not discovered:
            raise ValueError(f"No dataset resolutions found under {dataset_path}")
        return discovered

    normalized = sorted(set(resolutions))
    if not normalized:
        raise ValueError("No resolutions selected")
    return normalized


def _config_dir_has_parquets(config_dirpath: Path) -> bool:
    return config_dirpath.exists() and any(config_dirpath.rglob("*.parquet"))


def _collect_local_config_names(export_dirpath: Path) -> set[str]:
    if not export_dirpath.exists():
        return set()

    config_names: set[str] = set()
    for child in export_dirpath.iterdir():
        if child.is_dir() and _parse_config_name(child.name) is not None and _config_dir_has_parquets(child):
            config_names.add(child.name)
    return config_names


def _read_existing_config_names(
    export_dirpath: Path, repo_id: str, split_names: tuple[str, ...], solver_names: list[str]
) -> set[str]:
    """Read compatible card declarations; this does not verify remote uploads."""
    readme_filepath = export_dirpath / "README.md"
    if not readme_filepath.exists():
        return set()

    try:
        lines = readme_filepath.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "---":
            raise ValueError("missing YAML frontmatter")
        end = lines.index("---", 1)
        metadata = yaml.safe_load("\n".join(lines[1:end]))
        configs = metadata.get("configs") if isinstance(metadata, dict) else None
        if not isinstance(configs, list) or not configs:
            raise ValueError("configs must be a nonempty list")

        config_names: set[str] = set()
        for entry in configs:
            name = entry.get("config_name") if isinstance(entry, dict) else None
            parsed = _parse_config_name(name) if isinstance(name, str) else None
            if parsed is None or parsed[0] <= 0 or _config_name(*parsed) != name or name in config_names:
                raise ValueError(f"invalid or duplicate PASSAGE config name: {name!r}")
            expected_files = [{"split": split, "path": f"{name}/{split}/*"} for split in split_names]
            data_files = entry.get("data_files")
            if (
                not isinstance(data_files, list)
                or len(data_files) != len(expected_files)
                or any(data_files.count(expected) != 1 for expected in expected_files)
            ):
                raise ValueError(f"incompatible data_files for {name}")
            config_names.add(name)

        dataset_info = metadata.get("dataset_info")
        if not isinstance(dataset_info, list) or len(dataset_info) != len(config_names):
            raise ValueError("dataset_info must describe every retained config")
        current_header = _generate_huggingface_yaml_header(repo_id, config_names, list(split_names), None, solver_names)
        current_info = yaml.safe_load(current_header.split("---", 2)[1])["dataset_info"]
        expected_features = {entry["config_name"]: entry["features"] for entry in current_info}
        seen_info: set[str] = set()
        for entry in dataset_info:
            name = entry.get("config_name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or name not in config_names or name in seen_info:
                raise ValueError(f"invalid or duplicate dataset_info config: {name!r}")
            if entry.get("features") != expected_features[name]:
                raise ValueError(f"retained feature schema differs from the current export schema for {name}")
            seen_info.add(name)

        # Generated cards have no dedicated repo-id field. Check their companion
        # metadata before prepare_export rewrites it for the requested repository.
        croissant_filepath = export_dirpath / "croissant.json"
        if croissant_filepath.exists():
            croissant = json.loads(croissant_filepath.read_text(encoding="utf-8"))
            expected_url = _dataset_repo_url(repo_id)
            if not isinstance(croissant, dict) or any(croissant.get(key) != expected_url for key in ("@id", "url")):
                raise ValueError("existing Croissant repository identity differs from the requested repo_id")
        return config_names
    except (ValueError, yaml.YAMLError, UnicodeError) as exc:
        raise ValueError(f"Cannot keep existing configs from {readme_filepath}: {exc}") from exc


def _resolve_prepared_resolutions(
    export_dirpath: Path,
    dataset_path: Path,
    resolutions: list[int] | None,
) -> list[int]:
    if resolutions is not None:
        return sorted(set(resolutions))

    local_config_names = _collect_local_config_names(export_dirpath)
    if local_config_names:
        return sorted(
            {parsed[0] for config_name in local_config_names if (parsed := _parse_config_name(config_name)) is not None}
        )

    return _resolve_dataset_resolutions(dataset_path, None)


def _selected_config_names(resolutions: list[int]) -> set[str]:
    config_names: set[str] = set()
    for resolution in sorted(set(resolutions)):
        config_names.add(_config_name(resolution, True))
        config_names.add(_config_name(resolution, False))
    return config_names


def _validate_prepared_configs(export_dirpath: Path, resolutions: list[int]) -> None:
    missing: list[str] = []
    for config_name in _sort_config_names(_selected_config_names(resolutions)):
        if not _config_dir_has_parquets(export_dirpath / config_name):
            missing.append(config_name)

    if missing:
        missing_display = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing prepared parquet configs under {export_dirpath}: {missing_display}. Run `passage compress` first."
        )


def _metadata_sort_key(metadata_filepath: Path) -> tuple[int, str, str, str]:
    stem = metadata_filepath.stem
    if stem.isdigit():
        return (int(stem), metadata_filepath.parent.parent.name, metadata_filepath.parent.name, metadata_filepath.name)
    return (2**31 - 1, metadata_filepath.parent.parent.name, metadata_filepath.parent.name, metadata_filepath.name)


def _iter_metadata_filepaths(metadata_root: Path, limit: int | None = None) -> tuple[list[Path], int]:
    """Return metadata file paths in deterministic sample-index order."""

    filepaths = sorted(metadata_root.glob("*/*/*.yaml"), key=_metadata_sort_key)
    if limit is not None:
        filepaths = filepaths[:limit]
    return filepaths, len(filepaths)


def _bounded_ordered_imap(pool: "mp.pool.Pool", func, iterable, max_inflight: int):
    """Order-preserving pool map with bounded in-flight results.

    Unlike ``pool.imap``, which enqueues worker results onto an unbounded
    internal queue, this helper keeps at most ``max_inflight`` tasks pending so
    the main process cannot be overwhelmed when row payloads are large (e.g.
    full tensor bytes for high-resolution exports).
    """

    pending: deque = deque()
    it = iter(iterable)
    for _ in range(max(max_inflight, 1)):
        try:
            arg = next(it)
        except StopIteration:
            break
        pending.append(pool.apply_async(func, (arg,)))
    while pending:
        result = pending.popleft().get()
        try:
            arg = next(it)
        except StopIteration:
            pass
        else:
            pending.append(pool.apply_async(func, (arg,)))
        yield result


def _load_hf_environment(env_path: Path | None = None, secret_path: Path | None = None) -> tuple[Path, Path]:
    project_root = _repo_root()

    if env_path is None:
        env_path = project_root / ".env"
    if secret_path is None:
        secret_path = project_root / ".env.secret"

    if env_path.exists():
        load_dotenv(env_path, override=True)
    if secret_path.exists():
        load_dotenv(secret_path, override=True)

    return env_path, secret_path


def _load_hf_credentials(
    env_path: Path | None = None,
    secret_path: Path | None = None,
) -> tuple[str, str, str]:
    _load_hf_environment(env_path=env_path, secret_path=secret_path)

    username = os.getenv("HF_USER_NAME", "")
    repo_name = os.getenv("HF_REPO_NAME", "")
    token = os.getenv("HF_TOKEN", "")

    if not username:
        raise ValueError("HF_USER_NAME not found in .env file")
    if not repo_name:
        raise ValueError("HF_REPO_NAME not found in .env file")
    if not token:
        raise ValueError("HF_TOKEN not found in .env.secret file")

    return username, repo_name, token


def _load_hf_token(
    env_path: Path | None = None,
    secret_path: Path | None = None,
) -> str:
    _load_hf_environment(env_path=env_path, secret_path=secret_path)

    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN not found in .env.secret file")
    return token


def _has_required_assets(assets_dir: Path) -> bool:
    """An assets dir is considered ready if at least banner.png is present."""
    return (assets_dir / "banner.png").exists()


def _has_calibration_artifacts(calibration_dir: Path) -> bool:
    return (calibration_dir / "calibrate.json").exists()


def _has_analysis_notebooks(notebooks_dir: Path) -> bool:
    return notebooks_dir.exists() and any(notebooks_dir.glob("*.ipynb"))


def _is_valid_executed_notebook(notebook_path: Path) -> bool:
    """Check that an executed notebook file exists and parses as JSON."""
    if not notebook_path.exists() or notebook_path.stat().st_size == 0:
        return False
    try:
        import nbformat

        nbformat.read(str(notebook_path), as_version=4)
        return True
    except Exception:
        return False


def _copy_export_assets(notebooks_path: Path, export_dirpath: Path, force: bool = False) -> str:
    assets_dir = export_dirpath / "assets"
    if not force and _has_required_assets(assets_dir):
        return "skip"
    if force and assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    for image_filepath in (notebooks_path.parent / "assets").glob("*.png"):
        shutil.copy2(image_filepath.resolve(), (assets_dir / image_filepath.name).resolve())
    return "run"


def _copy_calibration_assets(calibrate_path: Path, export_dirpath: Path, force: bool = False) -> str:
    calibration_dir = export_dirpath / "calibration"
    if not force and _has_calibration_artifacts(calibration_dir):
        return "skip"
    if force and calibration_dir.exists():
        shutil.rmtree(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    for filepath in calibrate_path.glob("*"):
        if ".bak" in filepath.name or filepath.is_dir():
            continue
        shutil.copy2(filepath.resolve(), (calibration_dir / filepath.name).resolve())
    return "run"


def _copy_analysis_notebooks(notebooks_path: Path, export_dirpath: Path, force: bool = False) -> str:
    """Preserve the complete recursive source inventory, including templates."""
    from passage.notebook_export import write_staged_bytes

    changed = False
    for notebook_filepath in sorted(notebooks_path.rglob("*.ipynb")):
        destination = export_dirpath / "notebooks" / notebook_filepath.relative_to(notebooks_path)
        content = notebook_filepath.read_bytes()
        if force or not destination.exists() or destination.read_bytes() != content:
            write_staged_bytes(destination, content)
            changed = True
    return "run" if changed else "skip"


def _strip_notebook_outputs(notebook_path: Path) -> bool:
    """Strip outputs and execution counts from a notebook in-place. Return True if modified."""
    try:
        import nbformat
    except Exception:
        logger.warning("nbformat not available; cannot strip notebook outputs")
        return False

    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except Exception as exc:
        logger.warning(f"Failed to parse {notebook_path.name} for stripping: {exc}")
        return False

    modified = False
    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                modified = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                modified = True

    if modified:
        from passage.notebook_export import write_staged_bytes

        write_staged_bytes(notebook_path, nbformat.writes(nb).encode())
    return modified


def _strip_analysis_notebook_outputs(export_dirpath: Path, force: bool = False) -> tuple[int, int]:
    """Strip outputs from copied analysis notebooks. Returns (stripped, scanned)."""
    notebooks_dir = export_dirpath / "notebooks"
    if not notebooks_dir.exists():
        return (0, 0)

    stripped = 0
    scanned = 0
    for notebook_filepath in sorted(notebooks_dir.rglob("*.ipynb")):
        if notebook_filepath.relative_to(notebooks_dir).parts[0] in {"paper", "vision"}:
            continue
        scanned += 1
        before = notebook_filepath.stat().st_size
        if _strip_notebook_outputs(notebook_filepath):
            after = notebook_filepath.stat().st_size
            saved = max(0, before - after)
            logger.info(f"  stripped outputs from notebooks/{notebook_filepath.name} (-{saved // 1024} KB)")
            stripped += 1
        elif force:
            # No-op if already stripped, but reported when forced for transparency
            logger.info(f"  notebooks/{notebook_filepath.name} already stripped")
    return (stripped, scanned)


def _write_gitattributes(export_dirpath: Path, force: bool = False) -> str:
    target = export_dirpath / ".gitattributes"
    if target.exists() and not force:
        if target.read_text(encoding="utf-8") == GITATTRIBUTES_CONTENT:
            return "skip"
    target.write_text(GITATTRIBUTES_CONTENT, encoding="utf-8")
    return "run"


def _audit_parquet_compression(export_dirpath: Path, parquet_size_warn_mb: int) -> tuple[int, list[Path], list[Path]]:
    """Return (total_files, non_zstd_files, oversized_files)."""
    total = 0
    non_zstd: list[Path] = []
    oversized: list[Path] = []
    threshold_bytes = parquet_size_warn_mb * 1024 * 1024
    for parquet in export_dirpath.rglob("*.parquet"):
        total += 1
        if parquet.stat().st_size > threshold_bytes:
            oversized.append(parquet)
        try:
            pf = pq.ParquetFile(parquet)
            row_group = pf.metadata.row_group(0)
            comps = {row_group.column(i).compression for i in range(row_group.num_columns)}
            if comps != {"ZSTD"}:
                non_zstd.append(parquet)
        except Exception as exc:
            logger.warning(f"  failed to read parquet metadata for {parquet}: {exc}")
            non_zstd.append(parquet)
    return total, non_zstd, oversized


def _report_sample_sizes(export_dirpath: Path, sample_size_warn_mb: int) -> list[tuple[str, int]]:
    """List (config_name, bytes) for *_samples configs that exceed the threshold."""
    threshold = sample_size_warn_mb * 1024 * 1024
    oversized: list[tuple[str, int]] = []
    for child in sorted(export_dirpath.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_config_name(child.name)
        if parsed is None or not parsed[1]:
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        if size > threshold:
            oversized.append((child.name, size))
    return oversized


def _validate_croissant(croissant_filepath: Path) -> tuple[bool, str]:
    """Validate croissant.json. Try mlcroissant if installed; fall back to JSON sanity check."""
    if not croissant_filepath.exists():
        return False, f"croissant.json missing at {croissant_filepath}"

    try:
        croissant = json.loads(croissant_filepath.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"croissant.json is not valid JSON: {exc}"

    required_keys = {"@context", "@type", "name", "description", "url", "license"}
    missing = required_keys - croissant.keys()
    if missing:
        return False, f"croissant.json missing keys: {sorted(missing)}"

    try:
        import mlcroissant  # noqa: F401
    except Exception:
        return True, "ok (basic JSON check; install `mlcroissant` for full validation)"

    try:
        from mlcroissant import Dataset as CroissantDataset

        CroissantDataset(jsonld=str(croissant_filepath))
        return True, "ok (mlcroissant)"
    except Exception as exc:
        return False, f"mlcroissant validation failed: {exc}"


def _validate_readme_card(readme_filepath: Path, expected_configs: set[str]) -> tuple[bool, str]:
    if not readme_filepath.exists():
        return False, f"README.md missing at {readme_filepath}"

    try:
        from huggingface_hub import DatasetCard
    except Exception:
        return True, "ok (huggingface_hub.DatasetCard unavailable; skipped)"

    try:
        card = DatasetCard.load(str(readme_filepath), repo_type="dataset")
    except Exception as exc:
        return False, f"DatasetCard parse failed: {exc}"

    declared = {entry.get("config_name") for entry in (card.data.configs or [])}
    missing = expected_configs - declared
    if missing:
        return False, f"README YAML missing configs: {sorted(missing)}"
    return True, f"ok ({len(declared)} configs declared)"


def _copy_root_metadata_file(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and not force:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src.resolve(), dst.resolve())
    return "run"


def _write_root_metadata_file(content: str, dst: Path, force: bool) -> str:
    if dst.exists() and not force and dst.read_text(encoding="utf-8") == content:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return "run"


def _copy_release_metadata_idempotent(
    export_dirpath: Path,
    repo_id: str,
    force: bool,
) -> dict[str, str]:
    """Copy/transform root metadata files, skipping any that already exist when not forced."""
    repo_root = _repo_root()
    dataset_url = _dataset_repo_url(repo_id)
    statuses: dict[str, str] = {}

    statuses["LICENSE"] = _copy_root_metadata_file(repo_root / "LICENSE", export_dirpath / "LICENSE", force)
    statuses["CHANGELOG.md"] = _copy_root_metadata_file(
        repo_root / "CHANGELOG.md", export_dirpath / "CHANGELOG.md", force
    )
    statuses["DATASET_LICENSE.md"] = _copy_root_metadata_file(
        repo_root / "DATASET_LICENSE.md", export_dirpath / "DATASET_LICENSE.md", force
    )

    croissant_text = (repo_root / "croissant.json").read_text(encoding="utf-8")
    statuses["croissant.json"] = _write_root_metadata_file(
        _customize_croissant(croissant_text, repo_id=repo_id),
        export_dirpath / "croissant.json",
        force,
    )

    datasheet_text = (repo_root / "DATASHEET.md").read_text(encoding="utf-8")
    datasheet_text = datasheet_text.replace("https://huggingface.co/datasets/thalesgroup/passage", dataset_url)
    statuses["DATASHEET.md"] = _write_root_metadata_file(datasheet_text, export_dirpath / "DATASHEET.md", force)

    citation_text = (repo_root / "CITATION.cff").read_text(encoding="utf-8")
    statuses["CITATION.cff"] = _write_root_metadata_file(citation_text, export_dirpath / "CITATION.cff", force)

    return statuses


def _export_row_budget(resolution: int, requested: int) -> int:
    """Bound materialized row groups by uncompressed tensor bytes, at least one scene."""
    if requested <= 0:
        raise ValueError("Export row/shard size must be positive")
    from passage.dataset import DatasetSource

    return min(
        requested,
        DatasetSource.batch_size_for(resolution, ["tensor"]),
        max(1, (128 * 1024 * 1024) // (resolution * resolution * 12)),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_scene_shards(
    context: ReleaseContext,
    resolution: int,
    sample: bool,
    split: str,
    metadata_paths: list[Path],
    output_path: Path,
    *,
    prefix: str = "data",
    cpus: int = 0,
    preview_root: Path | None = None,
    first_shard: int = 0,
    receipt_hashes: list[str] | None = None,
) -> list[dict]:
    """Write bounded Arrow groups while preserving configured rows per physical shard."""
    shard_size = context.chunk_sizes.get(resolution)
    if shard_size is None:
        raise ValueError(f"Missing export.chunk_size entry for resolution {resolution}")
    batch_size = _export_row_budget(resolution, shard_size)
    output_path.mkdir(parents=True, exist_ok=True)
    pool = None
    worker_count = min(max(cpus - 1, 1), len(metadata_paths), batch_size) if cpus else 0
    processor = partial(_process_meta_for_prepare, resolution, sample, context.solver_names, preview_root=preview_root)
    source = metadata_paths
    if receipt_hashes is not None:
        if len(receipt_hashes) != len(metadata_paths):
            raise ValueError("Release admission receipt roster mismatch")
        source = zip(metadata_paths, receipt_hashes)
        processor = partial(_process_admitted_meta, resolution, sample, context.solver_names, preview_root=preview_root)
    if worker_count:
        pool = mp.Pool(processes=worker_count)
        iterable = _bounded_ordered_imap(pool, processor, source, max_inflight=worker_count)
    else:
        iterable = map(processor, source)
    shards, rows = [], []
    writer = None
    shard_rows = 0
    shard_path = None

    def flush():
        nonlocal writer, shard_rows, shard_path, rows
        if not rows:
            return
        table = Dataset.from_list(rows, features=_features(sample, context.solver_names)).data.table
        if writer is None:
            shard_path = output_path / f"{prefix}-{first_shard + len(shards):05d}.parquet"
            writer = pq.ParquetWriter(
                shard_path, table.schema, compression="zstd", compression_level=context.compression_level
            )
        writer.write_table(table)
        shard_rows += len(rows)
        rows = []
        if shard_rows == shard_size:
            writer.close()
            shards.append(
                {
                    "path": shard_path.name,
                    "rows": shard_rows,
                    "sha256": _file_sha256(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                    "schema_sha256": hashlib.sha256(pq.read_schema(shard_path).serialize().to_pybytes()).hexdigest(),
                }
            )
            writer = None
            shard_rows = 0

    try:
        for row in iterable:
            rows.append(row)
            if len(rows) >= min(batch_size, shard_size - shard_rows):
                flush()
        flush()
        if writer is not None:
            writer.close()
            writer = None
            shards.append(
                {
                    "path": shard_path.name,
                    "rows": shard_rows,
                    "sha256": _file_sha256(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                    "schema_sha256": hashlib.sha256(pq.read_schema(shard_path).serialize().to_pybytes()).hexdigest(),
                }
            )
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if writer is not None:
            writer.close()
        if pool is not None:
            pool.close()
            pool.join()
    return shards


def _process_admitted_meta(
    resolution: int,
    sample: bool,
    solver_names: list[str],
    admitted: tuple[Path, str],
    *,
    preview_root: Path | None = None,
) -> dict:
    path, receipt_hash = admitted
    return _process_meta_for_prepare(
        resolution, sample, solver_names, path, preview_root=preview_root, verified_receipt_sha256=receipt_hash
    )


def _bounded_executor_map(executor: Executor | None, function, jobs, *, max_inflight: int):
    """Yield ordered compact results without an unbounded executor task/result queue."""
    if executor is None:
        yield from map(function, jobs)
        return
    iterator = iter(jobs)
    pending = deque()
    try:
        for _ in range(max_inflight):
            job = next(iterator, None)
            if job is None:
                break
            pending.append(executor.submit(function, job))
        while pending:
            result = pending.popleft().result()
            job = next(iterator, None)
            if job is not None:
                pending.append(executor.submit(function, job))
            yield result
    except BaseException:
        # Cancellation must also stop the currently awaited future, which has
        # already been popped from pending. Native work cannot cancel through a
        # Future; stop and join every owned process before staging is removed.
        from passage.successor_generation import _stop_executor

        for future in pending:
            future.cancel()
        _stop_executor(executor, cancel=True)
        raise
    finally:
        for future in pending:
            future.cancel()


def _admit_successor_scenes(job: tuple[Path, int, tuple[str, ...], list[int]]) -> list[dict]:
    """Validate a bounded scene roster locally; return receipt records, never tensors."""
    from passage.successor_generation import validate_scene_receipt

    raw, resolution, split_names, indices = job
    scenes = []
    for index in indices:
        found = [
            (split, raw / str(resolution) / split / "receipts" / f"{index:010d}.json")
            for split in split_names
            if (raw / str(resolution) / split / "receipts" / f"{index:010d}.json").is_file()
        ]
        if len(found) != 1:
            raise ValueError(f"Requested scene {index} requires exactly one terminal receipt")
        split, path = found[0]
        before = _file_sha256(path)
        receipt = validate_scene_receipt(path.parent.parent, index)
        after = path.read_bytes()
        if hashlib.sha256(after).hexdigest() != before or json.loads(after) != receipt:
            raise ValueError("Release scene receipt changed during admission")
        scenes.append(
            {
                "index": index,
                "split": split,
                "receipt": receipt,
                "receipt_file_sha256": before,
                "receipt_semantic_sha256": hashlib.sha256(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":"), default=str).encode()
                ).hexdigest(),
            }
        )
    return scenes


def _write_successor_shard(job: tuple) -> dict:
    """Own one physical shard, its bounded row groups, compression and byte hashes."""
    context, resolution, sample, split, paths, hashes, output, prefix, shard_number, previews = job
    shards = _stream_scene_shards(
        context,
        resolution,
        sample,
        split,
        paths,
        output,
        prefix=prefix,
        preview_root=previews,
        first_shard=shard_number,
        receipt_hashes=hashes,
    )
    if len(shards) != 1:
        raise ValueError("Release physical shard job produced an invalid shard count")
    return {
        **shards[0],
        "samples": sample,
        "split": split,
        "path": (Path(_config_name(resolution, sample)) / split / shards[0]["path"]).as_posix(),
    }


def _successor_export_workers(
    resolution: int, shard_size: int, executor: Executor | None, num_cpus: int, memory_budget_bytes: int
) -> tuple[int, int]:
    if isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus < 0:
        raise ValueError("num_cpus must be a nonnegative integer (0 is serial)")
    if isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int) or memory_budget_bytes <= 0:
        raise ValueError("Export memory budget must be a positive integer")
    group_rows = _export_row_budget(resolution, shard_size)
    # Compressed row payloads, Arrow conversion, decoded canonical terrain and
    # Python metadata all exist in the producer, without crossing process IPC.
    per_worker = group_rows * resolution**2 * 12 * 3 + resolution**2 * 96 + 64 * 1024**2
    if memory_budget_bytes < per_worker:
        raise ValueError("Export memory budget cannot contain one shard producer")
    capacity = getattr(executor, "_max_workers", num_cpus or 1) if executor is not None else num_cpus or 1
    requested = min(capacity, num_cpus) if num_cpus else capacity
    return max(1, min(requested, memory_budget_bytes // per_worker)), per_worker


def _write_split_parquets(context: ReleaseContext, resolution: int, sample: bool, split: str, cpus: int) -> int:
    split_dir = context.dataset_path / str(resolution) / split
    if not split_dir.exists():
        return 0
    metadata_paths, count = _iter_metadata_filepaths(split_dir / "metadata", context.sample_size if sample else None)
    if not count:
        return 0
    output = context.export_dirpath / _config_name(resolution, sample) / split
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{split}-export-", dir=output.parent) as directory:
        temporary = Path(directory)
        shards = _stream_scene_shards(context, resolution, sample, split, metadata_paths, temporary, cpus=cpus)
        manifest = {
            "schema_version": 1,
            "resolution": resolution,
            "split": split,
            "population": "samples" if sample else "core",
            "rows": count,
            "shards": shards,
            "config_sha256": _file_sha256(context.config_file),
        }
        (temporary / "export-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        backup = output.with_name(f".{split}.previous")
        if backup.exists():
            raise ValueError(f"Previous export backup requires review: {backup}")
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(temporary, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return len(shards)


def _successor_history_roster(scenes: list[dict], resolution: int) -> list[dict]:
    records = []
    for scene in scenes:
        for item in scene["receipt"].get("attempt_history", []):
            source = item["receipt"]
            if not re.fullmatch(rf"failed-attempts/{scene['index']:010d}/[0-9]{{4}}\.json", source):
                raise ValueError("Release failure history path identity mismatch")
            records.append(
                {
                    "path": f"{resolution}/{scene['split']}/{source}",
                    "source_path": source,
                    "split": scene["split"],
                    "index": scene["index"],
                    "sha256": item["sha256"],
                }
            )
    if len({item["path"] for item in records}) != len(records):
        raise ValueError("Duplicate release failure history receipt")
    return records


def _verify_successor_history(root: Path, scenes: list[dict], resolution: int, records: list[dict]) -> None:
    expected = _successor_history_roster(scenes, resolution)
    keys = ("path", "source_path", "split", "index", "sha256")
    if [{key: record.get(key) for key in keys} for record in records] != expected:
        raise ValueError("Release failure history roster mismatch")
    by_identity = {(scene["split"], scene["index"]): scene["receipt"] for scene in scenes}
    for record in records:
        path = (root / record["path"]).resolve()
        if root not in path.parents or not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise ValueError("Release failure history content hash mismatch")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError("Release failure history size mismatch")
        archived = json.loads(path.read_text())
        current = by_identity[(record["split"], record["index"])]
        history = current.get("attempt_history", [])
        position = next(i for i, item in enumerate(history) if item["receipt"] == record["source_path"])
        if (
            any(
                archived.get(key) != current.get(key)
                for key in ("schema_version", "dataset_version", "index", "split", "resolution", "provenance_id")
            )
            or archived.get("scene_status") != "failed"
            or archived.get("artifacts")
            or len(archived.get("variants", {})) != 6
            or any(v.get("status") != "failed" for v in archived["variants"].values())
            or archived.get("attempt_history", []) != history[:position]
            or archived.get("infrastructure_attempt", 1) != history[position].get("attempt", 1)
        ):
            raise ValueError("Release failure history archived identity mismatch")


def verify_successor_batch(export_root: str | Path, batch_id: str) -> dict:
    """Verify committed shard bytes before an orchestrator removes raw transient scenes."""
    root = Path(export_root).resolve()
    if not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", batch_id):
        raise ValueError("Invalid release batch identity")
    manifest_path = root / "release-receipts" / f"batch-{batch_id}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("batch_id") != batch_id or manifest.get("schema_version") != "passage-export-batch-v1":
        raise ValueError("Release batch manifest identity mismatch")
    indices = manifest.get("scene_indices", [])
    scenes = manifest.get("scenes", [])
    if (
        not indices
        or indices != sorted(set(indices))
        or [scene.get("index") for scene in scenes] != indices
        or batch_id != f"{manifest.get('resolution')}-{indices[0]:010d}-{indices[-1]:010d}"
    ):
        raise ValueError("Release batch scene identity mismatch")
    expected_core, expected_samples = Counter(), Counter()
    expected_scene_statuses, expected_variant_statuses = Counter(), Counter()
    for scene in scenes:
        receipt = scene["receipt"]
        if (
            receipt.get("index") != scene["index"]
            or receipt.get("split") != scene["split"]
            or receipt.get("resolution") != manifest["resolution"]
        ):
            raise ValueError("Release batch receipt identity mismatch")
        if receipt["scene_status"] == "complete":
            expected_core[scene["split"]] += 1
        expected_scene_statuses[f"{scene['split']}:{receipt['scene_status']}"] += 1
        if len(receipt.get("variants", {})) != 6:
            raise ValueError("Release batch variant denominator mismatch")
        for variant in receipt["variants"].values():
            expected_variant_statuses[f"{scene['split']}:{variant['status']}"] += 1
    if (
        manifest.get("requested_scenes") != len(indices)
        or manifest.get("requested_variants") != 6 * len(indices)
        or Counter(manifest.get("scene_status_counts", {})) != expected_scene_statuses
        or Counter(manifest.get("variant_status_counts", {})) != expected_variant_statuses
    ):
        raise ValueError("Release batch terminal status denominator mismatch")
    for split, ids in manifest["sample_ids"].items():
        expected_samples[split] = len(ids)
    actual_core, actual_samples = Counter(), Counter()
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if root not in path.parents or not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise ValueError(f"Release batch content hash mismatch: {record['path']}")
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != record["rows"]:
                raise ValueError("Release batch row count mismatch")
            if (
                "schema_sha256" in record
                and hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest() != record["schema_sha256"]
            ):
                raise ValueError("Release batch schema identity mismatch")
            (actual_samples if record["samples"] else actual_core)[record["split"]] += record["rows"]
            if not record["samples"] and any(name.startswith("image_") for name in parquet.schema_arrow.names):
                raise ValueError("Core release shard contains preview images")
    if actual_core != expected_core or actual_samples != expected_samples:
        raise ValueError("Release batch core/sample row denominator mismatch")
    if len(manifest["scenes"]) != len(manifest["scene_indices"]):
        raise ValueError("Release batch terminal denominator mismatch")
    for scene in manifest["scenes"]:
        receipt = scene["receipt"]
        content = json.dumps(receipt, sort_keys=True, separators=(",", ":"), default=str).encode()
        if hashlib.sha256(content).hexdigest() != scene["receipt_semantic_sha256"]:
            raise ValueError("Release scene receipt content mismatch")
    _verify_successor_history(root, scenes, manifest["resolution"], manifest.get("history_files", []))
    return {**manifest, "manifest_sha256": _file_sha256(manifest_path), "raw_cleanup_safe": True}


def export_successor_batch(
    context: ReleaseContext,
    resolution: int,
    scene_indices: list[int],
    *,
    batch_id: str | None = None,
    executor: Executor | None = None,
    num_cpus: int = 0,
    memory_budget_bytes: int = DEFAULT_EXPORT_MEMORY_BUDGET_BYTES,
) -> dict:
    """Commit one bounded ascending global-index batch with replay and cleanup evidence.

    ``context.dataset_path`` is the raw batch root (containing resolution/split);
    ``context.export_dirpath`` is the separate release root. Every
    requested index needs exactly one validated complete or terminal receipt.
    The returned receipt retains all raw scene receipts and verified shard hashes.
    No source scene is deleted. Preview selection accumulates across batches.
    A supplied executor belongs to the generation session and remains reusable
    after success. On failure its workers are stopped and joined before staging
    cleanup. Without one, ``num_cpus=0`` preserves serial execution; a positive
    count owns a pool for this call. The memory budget caps concurrent producer
    workspaces independently of the executor's generation capacity.
    """
    from passage.regenerate import _require_source_checkout

    _require_source_checkout(__file__)
    if not scene_indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in scene_indices):
        raise ValueError("Release scene indices must be a nonempty roster of nonnegative integers")
    if scene_indices != sorted(set(scene_indices)):
        raise ValueError("Release scene indices must be unique and ascending")
    if resolution not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError("Unsupported release resolution")
    shard_size = context.chunk_sizes.get(resolution)
    if shard_size is None:
        raise ValueError(f"Missing export.chunk_size entry for resolution {resolution}")
    worker_count, worker_bytes = _successor_export_workers(
        resolution, shard_size, executor, num_cpus, memory_budget_bytes
    )
    if executor is None and num_cpus:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as owned_executor:
            return export_successor_batch(
                context,
                resolution,
                scene_indices,
                batch_id=batch_id,
                executor=owned_executor,
                num_cpus=num_cpus,
                memory_budget_bytes=memory_budget_bytes,
            )
    expected_batch_id = f"{resolution}-{scene_indices[0]:010d}-{scene_indices[-1]:010d}"
    if batch_id is not None and batch_id != expected_batch_id:
        raise ValueError("Batch identity must match resolution and global index bounds")
    batch_id = expected_batch_id
    root = context.export_dirpath.resolve()
    raw = context.dataset_path.resolve()
    if root == raw or raw in root.parents or root in raw.parents:
        raise ValueError("Raw batch and release must be separate trees")
    config_hash = _file_sha256(context.config_file)
    calibration_hash = _file_sha256(context.calibration_file)
    package_root = Path(__file__).resolve().parent.parent
    code_hashes = {
        p.relative_to(package_root).as_posix(): _file_sha256(p)
        for directory in (package_root / "passage", package_root / "passage-eval")
        for p in directory.rglob("*.py")
    }
    for path in (package_root.parent / "pyproject.toml", package_root.parent / "uv.lock"):
        if path.is_file():
            code_hashes[path.name] = _file_sha256(path)
    manifest_path = root / "release-receipts" / f"batch-{batch_id}.json"
    if manifest_path.is_file():
        previous = verify_successor_batch(root, batch_id)
        if (
            previous["scene_indices"] != scene_indices
            or previous["config_sha256"] != config_hash
            or previous["calibration_sha256"] != calibration_hash
            or previous["producer_code_sha256"] != code_hashes
        ):
            raise ValueError("Release batch resume provenance mismatch")
        _advance_successor_export_state(root, previous)
        return previous
    root.mkdir(parents=True, exist_ok=True)
    sample_ids = {split: [] for split in context.split_names}
    previous_manifests = {}
    state_path = root / "release-receipts" / f"state-{resolution}.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        previous_path = root / "release-receipts" / state["last_manifest"]
        if _file_sha256(previous_path) != state["last_manifest_sha256"]:
            raise ValueError("Release continuation manifest content mismatch")
        if (
            state["config_sha256"] != config_hash
            or state["calibration_sha256"] != calibration_hash
            or state["producer_code_sha256"] != code_hashes
        ):
            raise ValueError("Release provenance changed between batches")
        if state["last_index"] >= scene_indices[0]:
            raise ValueError("Release batches must advance global indices without overlap")
        previous_manifests[state["last_manifest"]] = state["last_manifest_sha256"]
        sample_ids.update(state["sample_ids_total"])
    elif any((root / "release-receipts").glob(f"batch-{resolution}-*.json")):
        raise ValueError("Release continuation state missing; resume the last committed batch first")
    if not state_path.is_file():
        for sample in (False, True):
            config_path = root / _config_name(resolution, sample)
            existing = [p for p in config_path.glob("*/*.parquet") if not p.name.startswith(f"data-{batch_id}-")]
            if existing:
                raise ValueError("Release export refuses untracked or mixed-revision Parquet shards")
    scenes, grouped = [], {split: [] for split in context.split_names}
    denominator, variants = Counter(), Counter()
    provenance_ids = set()
    expected_variants = {f"{solver}:{variant}" for solver in context.solver_names for variant in ("free", "obstacles")}
    admission_jobs = (
        (raw, resolution, context.split_names, scene_indices[start : start + 128])
        for start in range(0, len(scene_indices), 128)
    )
    for admitted in _bounded_executor_map(executor, _admit_successor_scenes, admission_jobs, max_inflight=worker_count):
        for scene in admitted:
            split, receipt = scene["split"], scene["receipt"]
            if (
                receipt.get("resolution") != resolution
                or receipt.get("split") != split
                or not receipt.get("provenance_id")
            ):
                raise ValueError("Release scene receipt identity or provenance mismatch")
            if receipt["scene_status"] == "complete" and receipt.get("calibration_sha256") != calibration_hash:
                raise ValueError("Release scene calibration identity mismatch")
            if set(receipt["variants"]) != expected_variants:
                raise ValueError("Release terminal variant solver identity mismatch")
            provenance_ids.add(receipt["provenance_id"])
            if len(provenance_ids) != 1:
                raise ValueError("Release batch contains mixed generation provenance")
            if state_path.is_file() and receipt["provenance_id"] != state["generation_provenance_id"]:
                raise ValueError("Release generation provenance changed between batches")
            denominator[f"{split}:{receipt['scene_status']}"] += 1
            for variant in receipt["variants"].values():
                variants[f"{split}:{variant['status']}"] += 1
            scenes.append(scene)
            if receipt["scene_status"] == "complete":
                metadata_path = raw / str(resolution) / split / receipt["artifacts"]["metadata"]["path"]
                grouped[split].append((metadata_path, scene["receipt_file_sha256"]))
    files, new_samples = [], {split: [] for split in context.split_names}
    history_files = _successor_history_roster(scenes, resolution)
    with tempfile.TemporaryDirectory(prefix=f".batch-{batch_id}-", dir=root) as temporary:
        temporary_root = Path(temporary)
        for record in history_files:
            source = raw / str(resolution) / record["split"] / record["source_path"]
            destination = temporary_root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            record["size_bytes"] = destination.stat().st_size
        _verify_successor_history(temporary_root, scenes, resolution, history_files)

        def shard_jobs():
            for split, paths in grouped.items():
                remaining = max(0, min(context.sample_size, 64) - len(sample_ids[split]))
                for sample, selected in ((False, paths), (True, paths[:remaining])):
                    if sample:
                        new_samples[split] = [path.stem for path, _ in selected]
                    for start in range(0, len(selected), shard_size):
                        roster = selected[start : start + shard_size]
                        relative = Path(_config_name(resolution, sample)) / split
                        yield (
                            context,
                            resolution,
                            sample,
                            split,
                            [path for path, _ in roster],
                            [digest for _, digest in roster],
                            temporary_root / relative,
                            f"data-{batch_id}",
                            start // shard_size,
                            temporary_root / "previews" / split,
                        )

        files.extend(_bounded_executor_map(executor, _write_successor_shard, shard_jobs(), max_inflight=worker_count))
        # On interruption an uncommitted deterministic shard is replaced on retry.
        for record in files + history_files:
            destination = root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_root / record["path"], destination)
    manifest = {
        "schema_version": "passage-export-batch-v1",
        "batch_id": batch_id,
        "resolution": resolution,
        "scene_indices": scene_indices,
        "scenes": scenes,
        "scene_status_counts": dict(denominator),
        "variant_status_counts": dict(variants),
        "requested_scenes": len(scene_indices),
        "requested_variants": len(scene_indices) * len(context.solver_names) * 2,
        "files": files,
        "history_files": history_files,
        "sample_ids": new_samples,
        "sample_ids_total": {split: sample_ids[split] + new_samples[split] for split in context.split_names},
        "sample_size_per_split": min(context.sample_size, 64),
        "config_sha256": config_hash,
        "calibration_sha256": calibration_hash,
        "generation_provenance_id": next(iter(provenance_ids)),
        "producer_code_sha256": code_hashes,
        "previous_manifest_sha256": previous_manifests,
    }
    manifest["execution"] = {
        "producer_max_inflight": worker_count,
        "producer_workspace_estimate_bytes": worker_bytes,
        "memory_budget_bytes": memory_budget_bytes,
        "row_group_rows": _export_row_budget(resolution, shard_size),
    }
    from passage.notebook_export import write_staged_bytes

    write_staged_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2, default=str) + "\n").encode())
    verified = verify_successor_batch(root, batch_id)
    _advance_successor_export_state(root, verified)
    return verified


def _advance_successor_export_state(root: Path, manifest: dict) -> None:
    """Keep bounded continuation state; complete batch manifests remain the evidence ledger."""
    path = root / "release-receipts" / f"state-{manifest['resolution']}.json"
    if path.is_file():
        state = json.loads(path.read_text())
        if state["last_index"] >= manifest["scene_indices"][-1]:
            return
    state = {
        "schema_version": "passage-export-state-v1",
        "resolution": manifest["resolution"],
        "last_index": manifest["scene_indices"][-1],
        "last_manifest": f"batch-{manifest['batch_id']}.json",
        "last_manifest_sha256": manifest["manifest_sha256"],
        "sample_ids_total": manifest["sample_ids_total"],
        **{
            key: manifest[key]
            for key in ("config_sha256", "calibration_sha256", "producer_code_sha256", "generation_provenance_id")
        },
    }
    from passage.notebook_export import write_staged_bytes

    write_staged_bytes(path, (json.dumps(state, sort_keys=True, indent=2) + "\n").encode())


def _write_export_readme(
    context: ReleaseContext,
    repo_id: str,
    config_names: set[str] | list[str],
    force: bool = False,
) -> str:
    if context.notebooks_path is None:
        raise ValueError("Notebooks path is required to build the export README")

    default_config_name = _select_default_config(config_names, context.configured_default)
    assets_readme_path = context.notebooks_path.parent / "assets" / "README.md"
    readme_text = _load_and_customize_readme(
        assets_readme_path,
        repo_id,
        config_names,
        list(context.split_names),
        default_config_name,
        context.solver_names,
    )
    return _write_root_metadata_file(readme_text, context.export_dirpath / "README.md", force)


def _upload_export_dir(
    api: HfApi,
    export_dirpath: Path,
    repo_id: str,
    upload_allow_patterns: list[str] | None,
    num_workers: int,
) -> None:
    max_retries = int(os.getenv("HF_UPLOAD_RETRIES", "3"))
    base_delay = float(os.getenv("HF_UPLOAD_RETRY_DELAY", "5"))
    max_delay = float(os.getenv("HF_UPLOAD_MAX_DELAY", "60"))

    attempt = 0
    while True:
        attempt += 1
        try:
            api.upload_large_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(export_dirpath.resolve()),
                allow_patterns=upload_allow_patterns,
                num_workers=num_workers,
                ignore_patterns=["*.ipynb_checkpoints"],
            )
            logger.info(f"✓ Uploaded export tree to https://huggingface.co/datasets/{repo_id}")
            return
        except (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.HTTPError) as exc:
            if attempt >= max_retries:
                logger.error(f"Upload failed after {attempt} attempts: {exc}")
                raise
            sleep_for = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(f"Upload timed out (attempt {attempt}/{max_retries}); retrying in {sleep_for}s: {exc}")
            time.sleep(sleep_for)


def _reset_hf_upload_cache(export_dirpath: Path) -> None:
    """Remove local Hugging Face upload cache for a clean upload pass."""
    cache_dir = export_dirpath / ".cache" / "huggingface"
    logger.info(f"Resetting Hugging Face upload cache at {cache_dir}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def compress_release(
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    notebooks_dirpath: str | Path | None = None,
    config_filepath: str | Path = "config/config.yaml",
    export_dirpath: str | Path | None = None,
    resolutions: list[int] | None = None,
    num_cpus: int | None = None,
) -> bool:
    """Generate Parquet shards for the selected resolutions."""

    try:
        context = _build_release_context(
            dataset_dirpath=dataset_dirpath,
            calibrate_dirpath=calibrate_dirpath,
            calibration_filepath=calibration_filepath,
            notebooks_dirpath=notebooks_dirpath,
            config_filepath=config_filepath,
            export_dirpath=export_dirpath,
            require_calibration=False,
            require_notebooks=False,
        )
        context.export_dirpath.mkdir(parents=True, exist_ok=True)

        selected_resolutions = _resolve_dataset_resolutions(context.dataset_path, resolutions)

        cpus = _normalize_cpus(num_cpus)
        produced_config_names: set[str] = set()
        with tempfile.TemporaryDirectory(prefix=".compression-", dir=context.export_dirpath) as directory:
            temporary_context = replace(context, export_dirpath=Path(directory))
            for resolution in selected_resolutions:
                for sample in [True, False]:
                    wrote_any_split = False
                    for split in context.split_names:
                        num_chunks = _write_split_parquets(temporary_context, resolution, sample, split, cpus)
                        wrote_any_split = wrote_any_split or num_chunks > 0
                    if wrote_any_split:
                        produced_config_names.add(_config_name(resolution, sample))
            if not produced_config_names:
                raise FileNotFoundError(
                    f"No metadata files found for selected resolutions under {context.dataset_path}"
                )
            backups, committed = {}, []
            try:
                for config_name in _sort_config_names(produced_config_names):
                    destination = context.export_dirpath / config_name
                    backup = context.export_dirpath / f".{config_name}.previous"
                    if backup.exists():
                        raise ValueError(f"Previous compression backup requires review: {backup}")
                    if destination.exists():
                        os.replace(destination, backup)
                        backups[destination] = backup
                    os.replace(Path(directory) / config_name, destination)
                    committed.append(destination)
            except BaseException:
                for destination in committed:
                    shutil.rmtree(destination)
                for destination, backup in backups.items():
                    os.replace(backup, destination)
                raise
            for backup in backups.values():
                shutil.rmtree(backup)

        logger.info("Prepared parquet configs: " + ", ".join(_sort_config_names(produced_config_names)))
        return True
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False


def prepare_export(
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    notebooks_dirpath: str | Path = "notebooks",
    config_filepath: str | Path = "config/config.yaml",
    export_dirpath: str | Path | None = None,
    resolutions: list[int] | None = None,
    repo_id: str | None = None,
    execute_notebooks: bool = True,
    max_notebook_retries: int | None = 3,
    notebook_retry_delay: int = 5,
    num_cpus: int | None = None,
    steps: ExportSteps | None = None,
    keep_existing_configs: bool = False,
    notebook_cell_timeout: int = 300,
    notebook_job_timeout: int = 1800,
    dataset_revision: str | None = None,
    result_root: str | Path | None = None,
) -> bool:
    """Prepare the export directory for upload.

    Idempotent by default: each step inspects the current export directory and skips work
    that already produced its expected artifact. Use ``ExportSteps.force_*`` flags (or the
    master ``force_all``) to redo a specific step. Notebook execution covers all four templates at each selected resolution. Outputs
    are reused only when their input/source/config identities and execution evidence match.

    ``keep_existing_configs`` also retains compatible declarations from the existing
    root README. Retain that card only after verifying earlier remote uploads; this
    option does not contact the repository or verify that those Parquets exist remotely.
    """

    steps = steps or ExportSteps()

    try:
        context = _build_release_context(
            dataset_dirpath=dataset_dirpath,
            calibrate_dirpath=calibrate_dirpath,
            calibration_filepath=calibration_filepath,
            notebooks_dirpath=notebooks_dirpath,
            config_filepath=config_filepath,
            export_dirpath=export_dirpath,
            require_calibration=True,
            require_notebooks=execute_notebooks,
        )

        selected_resolutions = _resolve_prepared_resolutions(context.export_dirpath, context.dataset_path, resolutions)
        _validate_prepared_configs(context.export_dirpath, selected_resolutions)

        repo_id_resolved = repo_id or DEFAULT_REPO_ID
        export_dir = context.export_dirpath
        existing_config_names = (
            _read_existing_config_names(export_dir, repo_id_resolved, context.split_names, context.solver_names)
            if keep_existing_configs
            else set()
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        successor = (
            any((export_dir / "release-receipts").glob("batch-*.json"))
            or (export_dir / "release-receipts/release-verification.json").is_file()
        )
        if successor and not execute_notebooks:
            raise ValueError("Release publication requires executed publication notebooks")
        logger.info(f"Preparing export at {export_dir}")
        logger.info(f"Selected resolutions: {selected_resolutions}")

        # Step 1 — assets
        status = _copy_export_assets(context.notebooks_path, export_dir, force=steps.force("force_assets"))
        logger.info(f"[1/9] assets {'(refreshed)' if status == 'run' else '(already present, skipped)'}")

        # Step 2 — calibration files
        status = _copy_calibration_assets(context.calibrate_path, export_dir, force=steps.force("force_calibration"))
        logger.info(f"[2/9] calibration {'(refreshed)' if status == 'run' else '(already present, skipped)'}")

        # Step 3 — analysis notebooks (raw copy)
        status = _copy_analysis_notebooks(
            context.notebooks_path, export_dir, force=steps.force("force_analysis_notebooks")
        )
        logger.info(f"[3/9] analysis notebooks {'(refreshed)' if status == 'run' else '(already present, skipped)'}")

        # Step 4 — strip outputs from analysis notebooks
        if steps.skip_strip_outputs:
            logger.info("[4/9] strip notebook outputs (disabled)")
        else:
            stripped, scanned = _strip_analysis_notebook_outputs(export_dir, force=steps.force("force_strip_outputs"))
            logger.info(f"[4/9] strip notebook outputs ({stripped}/{scanned} touched)")

        # Step 5 — root metadata (LICENSE, croissant, datasheet, citation, ...)
        statuses = (
            {}
            if successor
            else _copy_release_metadata_idempotent(export_dir, repo_id_resolved, force=steps.force("force_metadata"))
        )
        refreshed = [name for name, s in statuses.items() if s == "run"]
        skipped = [name for name, s in statuses.items() if s == "skip"]
        logger.info(
            "[5/9] root metadata "
            + (f"(refreshed: {', '.join(refreshed)})" if refreshed else "")
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )

        # Step 6 — complete matrix with identities and execution receipts.
        from passage.notebook_export import prepare_snapshot_notebooks, validate_publication_notebooks

        if max_notebook_retries is None or max_notebook_retries <= 0:
            raise ValueError("Publication notebook retries must be finite and positive")

        def execute_publication(input_path, output_path, parameters, **options):
            if not _execute_notebook_with_retries(
                input_path,
                output_path,
                parameters,
                cwd=options["cwd"],
                max_retries=max_notebook_retries,
                retry_delay=notebook_retry_delay,
                cell_timeout=options["cell_timeout"],
                job_timeout=options["job_timeout"],
            ):
                raise RuntimeError(f"Notebook execution failed: {input_path.name}")

        notebook_manifest = prepare_snapshot_notebooks(
            export_dir,
            export_dir,
            notebooks_dirpath=context.notebooks_path,
            resolutions=selected_resolutions,
            repo_id=repo_id_resolved,
            revision=dataset_revision,
            execute=execute_notebooks,
            cell_timeout=notebook_cell_timeout,
            job_timeout=notebook_job_timeout,
            config_filepath=context.config_file,
            result_root=result_root,
            force=steps.force("force_notebooks"),
            parameters={
                "MAX_WORKERS": _normalize_cpus(num_cpus),
                "CONFIG_FILEPATH": str(context.config_file),
                "CALIBRATE_FILEPATH": str(context.calibration_file),
            },
            _allow_prepared_root=True,
            executor=execute_publication,
        )
        if execute_notebooks:
            notebook_manifest["publication_admission"] = validate_publication_notebooks(
                notebook_manifest, export_dir, resolutions=selected_resolutions
            )
            from passage.notebook_export import MANIFEST_NAME, write_staged_bytes

            write_staged_bytes(
                export_dir / MANIFEST_NAME,
                (json.dumps(notebook_manifest, sort_keys=True, indent=2, default=str) + "\n").encode(),
            )
        executed = sum(e["execution"]["status"] == "executed" for e in notebook_manifest["notebooks"])
        logger.info(f"[6/9] notebook matrix ({executed} executed, complete source inventory copied)")

        # Step 7 — README (depends on prepared configs and per-resolution notebook layout)
        prepared_config_names = _collect_local_config_names(export_dir)
        if not prepared_config_names:
            raise FileNotFoundError(
                f"No prepared parquet configs found under {export_dir}. Run `passage compress` first."
            )
        prepared_config_names |= existing_config_names
        if successor:
            from passage.successor_publication import write_successor_publication_metadata

            write_successor_publication_metadata(context, repo_id=repo_id_resolved)
            status = "run"
        else:
            status = _write_export_readme(
                context, repo_id_resolved, prepared_config_names, force=steps.force("force_readme")
            )
        logger.info(f"[7/9] README.md {'(refreshed)' if status == 'run' else '(unchanged, skipped)'}")

        # Step 8 — .gitattributes (LFS rules)
        if steps.skip_gitattributes:
            logger.info("[8/9] .gitattributes (disabled)")
        else:
            status = _write_gitattributes(export_dir, force=steps.force("force_gitattributes"))
            logger.info(f"[8/9] .gitattributes {'(written)' if status == 'run' else '(unchanged, skipped)'}")

        # Step 9 — readiness validation (compression audit, sample-size warning, croissant + README)
        warnings: list[str] = []

        if not steps.skip_audit_compression:
            total, non_zstd, oversized = _audit_parquet_compression(export_dir, steps.parquet_size_warn_mb)
            if non_zstd:
                warnings.append(f"non-ZSTD parquet files ({len(non_zstd)}): {non_zstd[0]} ...")
            if oversized:
                warnings.append(
                    f"oversized parquet files (>{steps.parquet_size_warn_mb} MB, {len(oversized)}): "
                    + ", ".join(p.relative_to(export_dir).as_posix() for p in oversized[:3])
                )
            logger.info(
                f"[9/9] compression audit ({total} parquet files, {len(non_zstd)} non-ZSTD, {len(oversized)} oversized)"
            )
        else:
            logger.info("[9/9] compression audit (disabled)")

        oversized_samples = _report_sample_sizes(export_dir, steps.sample_size_warn_mb)
        if oversized_samples:
            for name, size in oversized_samples:
                warnings.append(
                    f"sample variant {name} is {size / (1024 * 1024):.1f} MB (>{steps.sample_size_warn_mb} MB)"
                )

        if not steps.skip_validate_croissant:
            ok, msg = _validate_croissant(export_dir / "croissant.json")
            (logger.info if ok else logger.warning)(f"  croissant validation: {msg}")
            if not ok:
                raise ValueError(f"croissant validation failed: {msg}")

        if not steps.skip_validate_readme:
            expected_configs = set(prepared_config_names)
            ok, msg = _validate_readme_card(export_dir / "README.md", expected_configs)
            (logger.info if ok else logger.warning)(f"  README YAML validation: {msg}")
            if not ok:
                raise ValueError(f"README YAML validation failed: {msg}")

        if warnings:
            logger.warning("Export prepared with warnings:")
            for w in warnings:
                logger.warning(f"  - {w}")
        logger.info(f"Prepared export directory at {export_dir}")
        if successor:
            from passage.successor_publication import verify_successor_publication_metadata

            verify_successor_publication_metadata(export_dir)
        return True
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return False


def upload_export(
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    export_dirpath: str | Path | None = None,
    resolutions: list[int] | None = None,
    repo_id: str | None = None,
    env_path: str | Path | None = None,
    secret_path: str | Path | None = None,
    token: str | None = None,
    private: bool = False,
    create_repo: bool = True,
    num_cpus: int | None = None,
) -> bool:
    """Upload the prepared export tree to Hugging Face."""

    try:
        resolved_export_dir = _resolve_export_dirpath(
            calibrate_dirpath=calibrate_dirpath,
            calibration_filepath=calibration_filepath,
            export_dirpath=export_dirpath,
        )
        if not resolved_export_dir.exists():
            raise FileNotFoundError(f"Export directory not found: {resolved_export_dir}")

        local_config_names = _collect_local_config_names(resolved_export_dir)
        if not local_config_names:
            raise FileNotFoundError(
                f"No prepared parquet configs found under {resolved_export_dir}. Run `passage compress` and `passage export` first."
            )

        upload_allow_patterns = None
        if resolutions is not None:
            selected_config_names = _selected_config_names(sorted(set(resolutions)))
            missing_config_names = sorted(selected_config_names - local_config_names)
            if missing_config_names:
                raise FileNotFoundError("Missing prepared configs for upload: " + ", ".join(missing_config_names))
            upload_allow_patterns = _build_incremental_upload_patterns(selected_config_names)

        resolved_token = token or _load_hf_token(
            env_path=Path(env_path) if env_path else None,
            secret_path=Path(secret_path) if secret_path else None,
        )
        resolved_repo_id = repo_id or DEFAULT_REPO_ID

        login(token=resolved_token)
        api = HfApi()
        if create_repo:
            api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=private, exist_ok=True)

        cpus = _normalize_cpus(num_cpus)
        upload_workers = 1 if cpus == 0 else max(cpus - 1, 1)
        _reset_hf_upload_cache(resolved_export_dir)
        _upload_export_dir(
            api=api,
            export_dirpath=resolved_export_dir,
            repo_id=resolved_repo_id,
            upload_allow_patterns=upload_allow_patterns,
            num_workers=upload_workers,
        )
        return True
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False


def export_and_upload(
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    calibration_filepath: str | Path | None = None,
    notebooks_dirpath: str | Path = "notebooks",
    config_filepath: str | Path = "config/config.yaml",
    resolutions: list[int] | None = None,
    repo_id: str | None = None,
    execute_notebooks: bool = True,
    max_notebook_retries: int | None = 3,
    notebook_retry_delay: int = 5,
    env_path: str | Path | None = None,
    secret_path: str | Path | None = None,
    private: bool = False,
    create_repo: bool = True,
    skip_upload: bool = False,
    num_cpus: int | None = None,
    export_dirpath: str | Path | None = None,
) -> bool:
    """Backward-compatible wrapper for the pre-split release flow."""

    resolved_repo_id = repo_id
    resolved_token: str | None = None
    if resolved_repo_id is None and not skip_upload:
        username, repo_name, resolved_token = _load_hf_credentials(
            env_path=Path(env_path) if env_path else None,
            secret_path=Path(secret_path) if secret_path else None,
        )
        resolved_repo_id = f"{username}/{repo_name}"

    resolved_repo_id = resolved_repo_id or DEFAULT_REPO_ID

    ok = compress_release(
        dataset_dirpath=dataset_dirpath,
        calibrate_dirpath=calibrate_dirpath,
        calibration_filepath=calibration_filepath,
        notebooks_dirpath=notebooks_dirpath,
        config_filepath=config_filepath,
        export_dirpath=export_dirpath,
        resolutions=resolutions,
        num_cpus=num_cpus,
    )
    if not ok:
        return False

    ok = prepare_export(
        dataset_dirpath=dataset_dirpath,
        calibrate_dirpath=calibrate_dirpath,
        calibration_filepath=calibration_filepath,
        notebooks_dirpath=notebooks_dirpath,
        config_filepath=config_filepath,
        export_dirpath=export_dirpath,
        resolutions=resolutions,
        repo_id=resolved_repo_id,
        execute_notebooks=execute_notebooks,
        max_notebook_retries=max_notebook_retries,
        notebook_retry_delay=notebook_retry_delay,
        num_cpus=num_cpus,
    )
    if not ok or skip_upload:
        return ok

    return upload_export(
        calibrate_dirpath=calibrate_dirpath,
        calibration_filepath=calibration_filepath,
        export_dirpath=export_dirpath,
        resolutions=resolutions,
        repo_id=resolved_repo_id,
        env_path=env_path,
        secret_path=secret_path,
        token=resolved_token,
        private=private,
        create_repo=create_repo,
        num_cpus=num_cpus,
    )


def _load_and_customize_readme(
    assets_readme_path: Path,
    repo_id: str,
    config_names: list[str] | set[str],
    split_names: list[str],
    default_config_name: str | None,
    solver_names: list[str],
) -> str:
    """Load README from assets and prepend Hugging Face YAML header with customizations."""

    # Generate the HF YAML header
    yaml_header = _generate_huggingface_yaml_header(
        repo_id, config_names, split_names, default_config_name, solver_names
    )

    # Load the base README from assets
    if assets_readme_path.exists():
        base_readme = assets_readme_path.read_text()
    else:
        logger.warning(f"Assets README not found at {assets_readme_path}, using minimal fallback")
        base_readme = "# PASSAGE Dataset\n\nElevation pathfinding dataset.\n"

    # Customize the repo_id in usage examples
    customized_readme = base_readme.replace("your-username/passage-dataset", repo_id)

    # Combine: YAML header + calibration stats + customized README
    return yaml_header + customized_readme
