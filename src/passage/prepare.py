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
import time
import subprocess

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
    """Write one Parquet shard with Arrow-typed columns (tensor, paths, image path).

    Tensors are stored as fixed-size nested lists with shape [resolution, resolution, 3]
    using Arrow's FixedSizeList type so high-dimensional arrays are preserved efficiently.
    Paths are stored as list<list<int32>> (variable-length sequences of pixel pairs).
    The `image` column contains a relative path to a PNG file included in the export (under `images/`), so
    Hugging Face Data Studio and `datasets.Image` can load and render the image by path.
    """

    if not rows:
        return

    # Fixed-size nested lists: [resolution][resolution][3] (float32)
    # Use nested list_ with explicit list_size for compatibility with older pyarrow
    tensor_type = pa.list_(pa.list_(pa.list_(pa.float32(), list_size=3), list_size=resolution), list_size=resolution)
    path_type = pa.list_(pa.list_(pa.int32()))

    ids = pa.array([r["id"] for r in rows], type=pa.string())
    images = pa.array([r.get("image") for r in rows], type=pa.binary())
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
    metadata = pa.array([r["metadata"] for r in rows])

    table = pa.Table.from_arrays(
        [ids, images, prefixes, tensors, path_free, path_obstacles, metadata],
        names=["id", "image", "prefix", "tensor", "path_free", "path_obstacles", "metadata"],
    )

    pq.write_table(table, shard_path, compression="zstd")


def _execute_notebook_with_papermill(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
) -> bool:
    """Execute a notebook with papermill and save the output.

    Prefer executing via the `uv run papermill` wrapper so that the same virtual
    environment used by `uv` is used for notebook execution. If that fails, fall
    back to using the imported `papermill` module when available.
    """

    # Build papermill CLI args: `uv run papermill input.ipynb output.ipynb -k python3 -p KEY VALUE ...`
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
        # papermill expects scalar values as strings
        cmd_base.extend(["-p", str(k), str(v)])

    # Try running via `uv run papermill` first
    try:
        logger.info(f"Running notebook with: {' '.join(cmd_base)}")
        subprocess.run(cmd_base, cwd=str(cwd) if cwd else None, check=True)
        logger.info(f"✓ Notebook executed (via uv run): {output_notebook.name}")
        return True
    except FileNotFoundError:
        logger.warning("`uv` not found in PATH; attempting local papermill execution if available")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"`uv run papermill` failed (exit {exc.returncode}); falling back to local papermill if available"
        )
    except KeyboardInterrupt:
        logger.warning(f"Notebook execution interrupted: {input_notebook.name}")
        return False

    # Fallback: try the imported papermill if present
    if PAPERMILL_AVAILABLE:
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
            logger.info(f"✓ Notebook executed (via papermill): {output_notebook.name}")
            return True
        except KeyboardInterrupt:
            logger.warning(f"Notebook execution interrupted: {input_notebook.name}")
            return False
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to execute notebook {input_notebook.name}: {exc}")
            return False

    logger.warning("Papermill execution not available; copying notebook without execution")
    return False


def _execute_notebook_with_retries(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
    max_retries: int | None = 3,
    retry_delay: int = 5,
) -> bool:
    """Execute a notebook using papermill, retrying until success or retries exhausted.

    - If `max_retries` is None, retries indefinitely until success (useful for CI where
      transient errors may occur). Returns True on successful execution and False when
      retries are exhausted.
    """

    attempt = 0
    while True:
        attempt += 1
        logger.info(f"Executing dataset notebook (attempt {attempt}) for resolution {parameters.get('RESOLUTION')}")
        success = _execute_notebook_with_papermill(input_notebook, output_notebook, parameters, cwd)
        if success:
            return True

        # Execution failed; decide whether to retry
        if max_retries is None:
            logger.warning("Notebook execution failed; retrying indefinitely (0 means infinite retries).")
            time.sleep(max(0, int(retry_delay)))
            continue

        if attempt >= max_retries:
            logger.warning(f"Notebook execution failed after {attempt} attempt(s).")
            return False

        time.sleep(max(0, int(retry_delay)))


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
- `image`: relative path to PNG file included in the export under `images/` (usable as `datasets.Image`)
- `prefix`: `<LAT>/<LON>` directory prefix
- `tensor`: float32 `[H, W, 3]` array (Arrow fixed-size lists)
- `path_free`: `[[i, j], ...]` path without obstacles
- `path_obstacles`: `[[i, j], ...]` path with obstacles
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
    max_notebook_retries: int | None = 3,
    notebook_retry_delay: int = 5,
) -> bool:
    """Prepare export artifacts in outputs/export (no upload).

    Notebook execution behavior:
    - If `execute_notebooks` is True, `dataset.ipynb` will be executed with papermill.
    - `max_notebook_retries` controls how many attempts to make before giving up and
      copying the notebook without execution. Set to 0 for unlimited retries.
    - `notebook_retry_delay` sets the delay (seconds) between retries.
    """

    export_path = Path(export_dirpath).resolve()
    dataset_path = Path(dataset_dirpath).resolve()

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False

    calibrate_path = Path(calibrate_dirpath).resolve()
    notebooks_path = Path(notebooks_dirpath).resolve()

    # Load export chunk sizes from config.yaml (hard-coded row counts per resolution)
    config_file = notebooks_path.parent / "config" / "config.yaml"
    chunk_map: dict[int, int] = {}
    try:
        if config_file.exists():
            cfg = yaml.safe_load(config_file.read_text()) or {}
            exp_cfg = cfg.get("export", {}) if isinstance(cfg, dict) else {}
            cs = exp_cfg.get("chunk_size", {}) if isinstance(exp_cfg, dict) else {}
            # Normalize keys to ints where possible
            for k, v in (cs or {}).items():
                try:
                    chunk_map[int(k)] = int(v)
                except Exception:
                    continue
        else:
            logger.warning(f"Config file not found at {config_file}; falling back to computed chunk sizes")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to parse config file {config_file}: {exc}")

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
                config_path = notebooks_path.parent / "config" / "config.yaml"
                calibrate_json_path = calibrate_path / "calibrate.json"
                max_workers_param = os.cpu_count() or 8

                parameters = {
                    "RESOLUTION": res,
                    "DATASET_DIRPATH": str(dataset_path),
                    "CONFIG_FILEPATH": str(config_path),
                    "CALIBRATE_FILEPATH": str(calibrate_json_path),
                    "MAX_WORKERS": max_workers_param,
                }

                # Interpret 0 as infinite retries for CLI convenience
                retries = None if max_notebook_retries == 0 else max_notebook_retries

                success = _execute_notebook_with_retries(
                    dataset_notebook,
                    output_notebook,
                    parameters=parameters,
                    cwd=notebooks_path,
                    max_retries=retries,
                    retry_delay=notebook_retry_delay,
                )

                if not success:
                    # Notebook could not be executed after retries; copy original without execution
                    shutil.copy2(dataset_notebook, output_notebook)

                    logger.warning(
                        "Notebook execution failed after configured retries; copied original without execution"
                    )
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

            # Determine shard size from configuration mapping when available, otherwise fall back to the heuristic
            shard_size = chunk_map.get(res) if chunk_map.get(res) is not None else _recommended_chunk_size(res)
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
                image_rel_path: str | None = None
                try:
                    if image_path.exists():
                        dest_image_dir = split_export / "images" / lat_dir / lon_dir
                        dest_image_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_image_dir / image_path.name
                        shutil.copy2(image_path, dest_file)
                        image_rel_path = str(Path("images") / lat_dir / lon_dir / image_path.name)
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Failed to copy image {image_path}: {exc}")
                    image_rel_path = None

                path_free, path_obstacles = _extract_paths(metadata)
                metadata_clean = _strip_paths_from_metadata(metadata)

                shard_rows.append(
                    {
                        "id": f"{idx:010d}",
                        "prefix": f"{lat_dir}/{lon_dir}",
                        "tensor": tensor,
                        "path_free": path_free,
                        "path_obstacles": path_obstacles,
                        "image": image_rel_path,
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

            logger.info(
                f"Split {split} ({res}): {total_rows} rows -> {shard_index} shard(s) (rows_per_shard={shard_size})"
            )

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
        f"  1. Create the dataset repo if needed:\n     uv run hf repo create {repo_id} --type dataset --exists-ok"
    )
    logger.info(
        "  2. Upload prepared export:\n"
        f"     cd {export_path} && uv run hf upload . {repo_id} --repo-type dataset --include '*'\n"
        f"     (or from elsewhere: uv run hf upload {export_path} {repo_id} --repo-type dataset --include '*')"
    )

    return True
