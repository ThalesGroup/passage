"""Prepare module for PASSAGE.

Creates a Hugging Face-friendly export structure under outputs/export with
chunked parquet shards per resolution and split, plus documentation notebooks.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger

try:
    import papermill as pm

    PAPERMILL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PAPERMILL_AVAILABLE = False
    pm = None


def _recommended_chunk_size(resolution: int, user_chunk_size: int | None = None) -> int:
    """Compute a Parquet shard size targeting ~500 MB per file."""

    if user_chunk_size is not None:
        return max(1, int(user_chunk_size))

    bytes_per_sample = resolution * resolution * 3 * 4  # float32 tensor
    target_bytes = 500 * 1024 * 1024
    estimated = max(1, int(target_bytes / bytes_per_sample))
    return max(1, min(512, estimated))


def _strip_paths_from_metadata(metadata: dict) -> dict:
    """Remove heavy pixel lists from metadata while keeping path stats."""

    meta = copy.deepcopy(metadata)
    paths = meta.get("paths")
    if isinstance(paths, dict):
        for key in ("without_obstacles", "with_obstacles"):
            if key in paths and isinstance(paths[key], dict):
                paths[key].pop("pixels", None)
    return meta


def _extract_paths(metadata: dict) -> tuple[list[list[int]], list[list[int]]]:
    """Extract pixel indices for free/obstacle paths as list[list[int, int]]."""

    def _to_pairs(path_dict: dict | None) -> list[list[int]]:
        if not path_dict or "pixels" not in path_dict:
            return []
        pixels = path_dict.get("pixels", [])
        pairs: list[list[int]] = []
        for item in pixels:
            try:
                pairs.append([int(item.get("i")), int(item.get("j"))])
            except Exception:
                continue
        return pairs

    paths = metadata.get("paths", {}) if isinstance(metadata, dict) else {}
    free_pairs = _to_pairs(paths.get("without_obstacles"))
    obstacles_pairs = _to_pairs(paths.get("with_obstacles"))
    return free_pairs, obstacles_pairs


def _write_parquet_shard(rows: list[dict], shard_path: Path, resolution: int) -> None:
    """Write one Parquet shard with Arrow-typed columns (tensor, paths, image).

    Tensors are stored as fixed-size nested lists with shape [resolution, resolution, 3]
    using Arrow's FixedSizeList type so high-dimensional arrays are preserved efficiently.
    Paths are stored as list<list<int32>> (variable-length sequences of pixel pairs).
    """

    if not rows:
        return

    # Fixed-size nested lists: [resolution][resolution][3] (float32)
    # Use nested list_ with explicit list_size for compatibility with older pyarrow
    tensor_type = pa.list_(pa.list_(pa.list_(pa.float32(), list_size=3), list_size=resolution), list_size=resolution)
    path_type = pa.list_(pa.list_(pa.int32()))

    ids = pa.array([r["id"] for r in rows], type=pa.string())
    prefixes = pa.array([r["prefix"] for r in rows], type=pa.string())

    # Normalize tensors to numpy.float32 and validate/reshape where reasonable
    tensors_np = []
    for r in rows:
        t = np.asarray(r["tensor"], dtype=np.float32)
        # If 2D -> expand to 3 channels
        if t.ndim == 2:
            t = np.stack([t, np.zeros_like(t, dtype=np.float32), np.zeros_like(t, dtype=np.float32)], axis=-1)
        # If 3D but wrong channel count, pad or truncate
        if t.ndim == 3:
            if t.shape[2] < 3:
                pad = np.zeros((t.shape[0], t.shape[1], 3 - t.shape[2]), dtype=np.float32)
                t = np.concatenate([t, pad], axis=2)
            elif t.shape[2] > 3:
                t = t[:, :, :3]
        # Ensure expected spatial dims match resolution when possible
        if t.shape[0] != resolution or t.shape[1] != resolution:
            # Try to reshape from flat array if possible
            if t.size == resolution * resolution * 3:
                t = t.reshape((resolution, resolution, 3))
            else:
                raise ValueError(f"Tensor shape mismatch: expected ({resolution},{resolution},3) got {t.shape}")
        tensors_np.append(t.tolist())

    tensors = pa.array(tensors_np, type=tensor_type)

    path_free = pa.array([r["path_free"] for r in rows], type=path_type)
    path_obstacles = pa.array([r["path_obstacles"] for r in rows], type=path_type)
    images = pa.array([r.get("image") for r in rows], type=pa.binary())
    metadata = pa.array([r["metadata"] for r in rows])

    table = pa.Table.from_arrays(
        [ids, prefixes, tensors, path_free, path_obstacles, images, metadata],
        names=["id", "prefix", "tensor", "path_free", "path_obstacles", "image", "metadata"],
    )

    pq.write_table(table, shard_path, compression="zstd")


def _execute_notebook_with_papermill(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
) -> bool:
    """Execute a notebook with papermill and save the output."""

    if not PAPERMILL_AVAILABLE:
        logger.warning("papermill not installed; copying notebook without execution")
        return False

    try:
        old_cwd = os.getcwd()
        if cwd:
            os.chdir(cwd)
        try:
            pm.execute_notebook(
                str(input_notebook.resolve()),
                str(output_notebook.resolve()),
                parameters=parameters,
                kernel_name="python3",
            )
        finally:
            os.chdir(old_cwd)
        logger.info(f"✓ Notebook executed: {output_notebook.name}")
        return True
    except KeyboardInterrupt:
        logger.warning(f"Notebook execution interrupted: {input_notebook.name}")
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to execute notebook {input_notebook.name}: {exc}")
        return False


def _generate_dataset_readme(
    repo_id: str,
    resolutions: list[int],
    split_names: list[str],
    calibrate_data: dict | None,
) -> str:
    """Generate a dataset README.md with config/split definitions."""

    configs_yaml = []
    for res in sorted(resolutions):
        split_entries = []
        for split in sorted(split_names):
            split_entries.append(f'      - split: {split}\n        path: "{res}/{split}/data-*.parquet"')
        configs_yaml.append('  - config_name: "%s"\n    data_files:\n%s' % (res, "\n".join(split_entries)))

    calibration_info = ""
    if calibrate_data:
        calibration_info = (
            f"- Global min elevation: {calibrate_data.get('global_min', 'N/A')}\n"
            f"- Global max elevation: {calibrate_data.get('global_max', 'N/A')}\n"
            f"- Tiles analyzed: {len(calibrate_data.get('tile_stats', []))}\n"
        )

    return f"""---
license: mit
task_categories:
  - image-segmentation
  - other
tags:
  - elevation
  - pathfinding
  - dsm
  - alos
  - passage
  - terrain
configs:
{os.linesep.join(configs_yaml)}
---

# PASSAGE Elevation Dataset (multi-resolution)

This repository contains a single dataset with multiple configs (one per resolution) and
four splits: train, calibration, validation, and test. Each sample is a small terrain
patch represented as a 3-channel float32 tensor plus metadata and optional PNG visualization.

## What this dataset is for
- **Task**: path finding on elevation maps. Each sample contains two representative
  pixel-level paths:
  - **`path_free`**: a path computed on the raw terrain without obstacles (baseline);
  - **`path_obstacles`**: a path computed on the same terrain with synthetic obstacles applied.
  Models can be trained to predict feasible paths, estimate cost differences, or reason about
  obstacle avoidance on elevation-based inputs.

## Where the data come from
- Source elevation tiles are taken from the ALOS AW3D30 DSM product (publicly available global
  digital surface model). Tiles are cropped around randomized centers and normalized using
  globally computed calibration statistics.

## How obstacles are generated
- Obstacles are synthesized procedurally during dataset generation. For each crop the generator:
  - decides a target obstacle ratio and sizes (configurable);
  - places obstacle regions (blobs) randomly while respecting size constraints and coverage targets;
  - marks obstacles in a dedicated channel and recomputes the path to obtain `path_obstacles`.
  The generated metadata contains obstacle statistics (`ratio_actual`, `ratio_target`, `size_min_ratio`, `size_max_ratio`, `penalty`) useful for analysis.

## Parquet schema
- `id`: sample index (zero-padded)
- `prefix`: `<LAT>/<LON>` directory prefix
- `tensor`: float32 `[H, W, 3]` array (Arrow fixed-size lists)
- `path_free`: `[[i, j], ...]` path without obstacles
- `path_obstacles`: `[[i, j], ...]` path with obstacles
- `image`: PNG bytes with both paths drawn (usable as `datasets.Image`)
- `metadata`: metadata without pixel-level path lists

## Calibration summary
{calibration_info if calibration_info else "(calibration data not included)"}

## Calibration maps
The following diagnostic maps are included under `calibrate/` in this export and are useful
for understanding the elevation distribution over tiles:

- `calibrate/max_elevation_map.png` — per-tile maximum elevation visualization
- `calibrate/ptp_elevation_map.png` — per-tile peak-to-peak (range) elevation visualization

## Usage

```python
from datasets import load_dataset

# Load one resolution config
# e.g., 256 for 256x256 samples

dataset = load_dataset("{repo_id}", "256")
print(dataset)
```

"""


def prepare_export(
    export_dirpath: str | Path = "outputs/export",
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    notebooks_dirpath: str | Path = "notebooks",
    resolutions: list[int] | None = None,
    repo_id: str = "username/passage",
    execute_notebooks: bool = True,
    max_samples_notebook: int | None = None,
    chunk_size: int | None = None,
) -> bool:
    """Prepare export artifacts in outputs/export (no upload)."""

    export_path = Path(export_dirpath).resolve()
    dataset_path = Path(dataset_dirpath).resolve()

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False

    calibrate_path = Path(calibrate_dirpath).resolve()
    notebooks_path = Path(notebooks_dirpath).resolve()

    if resolutions is None:
        resolutions = [int(d.name) for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        resolutions.sort()

    if not resolutions:
        logger.warning("No resolution directories found to prepare")
        return False

    export_path.mkdir(parents=True, exist_ok=True)

    calibrate_data = None
    calibrate_json = calibrate_path / "calibrate.json"
    if calibrate_json.exists():
        calibrate_data = json.loads(calibrate_json.read_text())

    # Copy notebooks (excluding dataset.ipynb)
    notebooks_export = export_path / "notebooks"
    notebooks_export.mkdir(exist_ok=True)
    if notebooks_path.exists():
        for nb in notebooks_path.glob("*.ipynb"):
            if nb.name != "dataset.ipynb":
                shutil.copy2(nb, notebooks_export / nb.name)

    # Execute dataset.ipynb per resolution
    dataset_notebook = notebooks_path / "dataset.ipynb"

    split_names = ["train", "calibration", "validation", "test"]

    for res in resolutions:
        res_export = export_path / str(res)
        res_export.mkdir(exist_ok=True)

        res_data = dataset_path / str(res)
        if not res_data.exists():
            logger.warning(f"Resolution {res} data not found at {res_data}, skipping")
            continue

        summary_json = res_data / "summary.json"
        if summary_json.exists():
            shutil.copy2(summary_json, res_export / "summary.json")

        if dataset_notebook.exists():
            output_notebook = res_export / "dataset.ipynb"
            if execute_notebooks:
                parameters = {
                    "RESOLUTION": res,
                    "MAX_SAMPLES": 0 if max_samples_notebook is None else max_samples_notebook,
                    "SAVE_WORLD_COVER": False,
                }
                if not _execute_notebook_with_papermill(dataset_notebook, output_notebook, parameters, notebooks_path):
                    shutil.copy2(dataset_notebook, output_notebook)
            else:
                shutil.copy2(dataset_notebook, output_notebook)

        # Generate parquet shards per split
        for split in split_names:
            split_src = res_data / split
            if not split_src.exists():
                continue

            tensors_root = split_src / "tensors"
            metadata_root = split_src / "metadata"

            split_export = res_export / split
            split_export.mkdir(parents=True, exist_ok=True)

            if not metadata_root.exists():
                logger.warning(f"No metadata for split {split} at {metadata_root}, skipping")
                continue

            shard_size = _recommended_chunk_size(res, chunk_size)
            shard_rows: list[dict] = []
            shard_index = 0
            total_rows = 0

            for meta_file in sorted(metadata_root.rglob("*.yaml")):
                try:
                    idx = int(meta_file.stem)
                except ValueError:
                    continue

                lat_dir = meta_file.parent.parent.name
                lon_dir = meta_file.parent.name
                tensor_path = tensors_root / lat_dir / lon_dir / f"{idx:010d}.npy"
                if not tensor_path.exists():
                    continue

                try:
                    metadata = yaml.safe_load(meta_file.read_text())
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Failed to parse metadata {meta_file}: {exc}")
                    continue
                if not isinstance(metadata, dict):
                    continue
                tensor = np.load(tensor_path).astype(np.float32)

                image_path = split_src / "images" / lat_dir / lon_dir / f"{idx:010d}.png"
                image_bytes: bytes | None
                try:
                    image_bytes = image_path.read_bytes() if image_path.exists() else None
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Failed to read image {image_path}: {exc}")
                    image_bytes = None

                path_free, path_obstacles = _extract_paths(metadata)
                metadata_clean = _strip_paths_from_metadata(metadata)

                shard_rows.append(
                    {
                        "id": f"{idx:010d}",
                        "prefix": f"{lat_dir}/{lon_dir}",
                        "tensor": tensor,
                        "path_free": path_free,
                        "path_obstacles": path_obstacles,
                        "image": image_bytes,
                        "metadata": metadata_clean,
                    }
                )
                total_rows += 1

                if len(shard_rows) >= shard_size:
                    shard_path = split_export / f"data-{shard_index:05d}.parquet"
                    _write_parquet_shard(shard_rows, shard_path, res)
                    shard_rows.clear()
                    shard_index += 1

            if shard_rows:
                shard_path = split_export / f"data-{shard_index:05d}.parquet"
                _write_parquet_shard(shard_rows, shard_path, res)
                shard_index += 1

            logger.info(f"Split {split} ({res}): {total_rows} rows -> {shard_index} shard(s) (chunk_size={shard_size})")

    # Copy calibration artifacts into the export so HF consumers have calibration stats and maps
    export_calibrate = export_path / "calibrate"
    export_calibrate.mkdir(parents=True, exist_ok=True)
    try:
        cal_json = calibrate_path / "calibrate.json"
        if cal_json.exists():
            shutil.copy2(cal_json, export_calibrate / "calibrate.json")
        for png in calibrate_path.glob("*.png"):
            try:
                shutil.copy2(png, export_calibrate / png.name)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed copying calibration image {png}: {e}")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to copy calibration artifacts: {exc}")

    readme = _generate_dataset_readme(repo_id, resolutions, split_names, calibrate_data)
    (export_path / "README.md").write_text(readme)

    logger.info(f"\n✓ Export prepared at {export_path}")
    logger.info("\nNext steps (Hugging Face Hub):")
    logger.info(
        "  1. Create the dataset repo if needed:\n"
        f"     uv run huggingface-cli repo create {repo_id} --type dataset --exists-ok"
    )
    logger.info(
        "  2. Upload prepared export:\n"
        f"     cd {export_path} && uv run huggingface-cli upload . {repo_id} --repo-type dataset --include '*'\n"
        f"     (or from elsewhere: uv run huggingface-cli upload {export_path} {repo_id} --repo-type dataset --include '*')"
    )

    return True
