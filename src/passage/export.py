"""Export module for PASSAGE.

This module provides functionality to export generated datasets to Hugging Face Hub.
Datasets are organized per resolution and include calibration data.

Two modes of operation:
1. `prepare` - Prepare export structure in outputs/Export/ (local, no upload)
2. `upload` - Upload prepared exports to Hugging Face Hub
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from loguru import logger

# Optional huggingface_hub import
try:
    from huggingface_hub import HfApi, create_repo, upload_folder

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HfApi = None
    create_repo = None
    upload_folder = None

# Optional papermill import
try:
    import papermill as pm

    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False
    pm = None


def _check_hf_available() -> bool:
    """Check if huggingface_hub is available."""
    if not HF_AVAILABLE:
        logger.error(
            "huggingface_hub is not installed. Install it with:\n"
            "  uv add huggingface_hub\n"
            "  # or\n"
            "  pip install huggingface_hub"
        )
        return False
    return True


def _check_hf_login() -> bool:
    """Check if user is logged in to Hugging Face."""
    try:
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"Logged in to Hugging Face as: {user_info.get('name', 'unknown')}")
        return True
    except Exception as e:
        logger.error(
            f"Not logged in to Hugging Face: {e}\n"
            "Please login with:\n"
            "  uv run hf auth login\n"
            "  # or set HF_TOKEN environment variable"
        )
        return False


def _generate_dataset_card(
    repo_id: str,
    resolution: int | None,
    calibrate_data: dict | None,
    summary_data: dict | None,
    is_calibration: bool = False,
) -> str:
    """Generate a README.md dataset card for Hugging Face."""
    if is_calibration:
        return f"""---
license: mit
task_categories:
  - other
tags:
  - elevation
  - dsm
  - alos
  - calibration
  - passage
size_categories:
  - 10K<n<100K
---

# PASSAGE Calibration Data

This dataset contains calibration data for the PASSAGE elevation dataset generator.

## Description

Calibration data extracted from ALOS AW3D30 DSM (Digital Surface Model) tiles,
including global min/max elevation values and per-tile statistics.

## Global Statistics

- **Global Min Elevation**: {calibrate_data.get("global_min", "N/A")} meters
- **Global Max Elevation**: {calibrate_data.get("global_max", "N/A")} meters
- **Total Tiles**: {len(calibrate_data.get("tile_stats", []))} tiles

## Files

- `calibrate.json`: Full calibration data including per-tile statistics
- `max_elevation_map.png`: Visualization of maximum elevations per tile
- `ptp_elevation_map.png`: Visualization of peak-to-peak elevation range per tile

## Usage

```python
import json
from huggingface_hub import hf_hub_download

# Download calibration data
calibrate_path = hf_hub_download(repo_id="{repo_id}", filename="calibrate.json", repo_type="dataset")
with open(calibrate_path) as f:
    calibration = json.load(f)

global_min = calibration["global_min"]
global_max = calibration["global_max"]
tile_stats = calibration["tile_stats"]
```

## Source

Generated using [PASSAGE](https://github.com/YOUR_USERNAME/passage) from ALOS AW3D30 DSM tiles.

## License

MIT License
"""
    else:
        num_samples = summary_data.get("num_samples", "N/A") if summary_data else "N/A"
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
size_categories:
  - 1K<n<10K
---

# PASSAGE Elevation Dataset - {resolution}x{resolution}

This dataset contains elevation data samples at {resolution}x{resolution} resolution for pathfinding tasks.

## Description

Elevation tensor samples generated from ALOS AW3D30 DSM (Digital Surface Model) tiles.
Each sample includes normalized elevation data, marker positions for start/end points,
obstacle masks, and pre-computed shortest paths.

## Dataset Statistics

- **Resolution**: {resolution}x{resolution} pixels
- **Number of Samples**: {num_samples}
- **Global Min Elevation**: {calibrate_data.get("global_min", "N/A") if calibrate_data else "N/A"} meters
- **Global Max Elevation**: {calibrate_data.get("global_max", "N/A") if calibrate_data else "N/A"} meters

## Files Structure

```
tensors/          # .npy files (H×W×3 float32)
  ├── N045/
  │   └── E010/
  │       └── 0000000000.npy
  └── ...
metadata/         # .yaml sample metadata
  └── ...
images/           # .png visualization (optional)
  └── ...
summary.json      # Dataset generation summary
```

## Tensor Format

Each `.npy` tensor is a `({resolution}, {resolution}, 3)` float32 array:
- **Channel 0**: Normalized elevation (0-1 range using global min/max)
- **Channel 1**: Marker channel (-1 everywhere, 0 at start, 1 at end)
- **Channel 2**: Obstacles mask (1.0 for blocked, 0.0 for free)

## Usage

```python
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

# Download a single tensor
tensor_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="tensors/N045/E010/0000000000.npy",
    repo_type="dataset"
)
tensor = np.load(tensor_path)
elevation = tensor[:, :, 0]
markers = tensor[:, :, 1]
obstacles = tensor[:, :, 2]

# Download entire dataset
local_dir = snapshot_download(repo_id="{repo_id}", repo_type="dataset")
```

## Metadata Format

Each `.yaml` metadata file contains:
- Sample index and resolution
- Geographic coordinates (center, top-left, bottom-right)
- Source tile information
- Marker positions with elevations
- Computed paths (with and without obstacles)
- Obstacle configuration
- Solver settings used

## Source

Generated using [PASSAGE](https://github.com/YOUR_USERNAME/passage) from ALOS AW3D30 DSM tiles.

## License

MIT License
"""


def _generate_multi_dataset_card(
    repo_id: str,
    resolutions: list[int],
    split_names: list[str],
    summaries: dict[int, dict] | None,
    calibrate_data: dict | None,
    include_images: bool = True,
) -> str:
    """Generate a README.md for a single HF repo containing all resolutions and splits."""

    summaries = summaries or {}
    split_names_sorted = sorted(split_names)
    configs_yaml = []
    for res in sorted(resolutions):
        split_entries = []
        for split in split_names_sorted:
            split_entries.append(f'      - split: {split}\n        path: "{res}/{split}/tensors/**/*.npy"')
        configs_yaml.append('  - config_name: "%s"\n    data_files:\n%s' % (res, "\n".join(split_entries)))

    total_samples = sum(int(s.get("successful_samples", 0)) for s in summaries.values())

    def _fmt_splits(summary: dict) -> str:
        splits = summary.get("splits", {}) if summary else {}
        parts = []
        for split in split_names_sorted:
            stats = splits.get(split, {})
            parts.append(
                f"- {split}: requested={stats.get('requested', 'N/A')}, successful={stats.get('successful', 'N/A')}"
            )
        return "\n".join(parts)

    per_res_blocks = []
    for res in sorted(resolutions):
        summary = summaries.get(res, {})
        per_res_blocks.append(
            f"### Resolution {res}×{res}\n"
            f"- Successful samples: {summary.get('successful_samples', 'N/A')} / {summary.get('requested_samples', 'N/A')}\n"
            f"- Global min/max: {summary.get('global_min', 'N/A')} / {summary.get('global_max', 'N/A')}\n"
            f"- Split breakdown:\n{_fmt_splits(summary)}\n"
        )

    images_note = "Images uploaded" if include_images else "Images excluded (tensors + metadata only)"

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
size_categories:
  - 1K<n<10K
---

# PASSAGE Elevation Dataset (multi-resolution)

Single repository containing PASSAGE elevation samples for multiple resolutions and splits (train/calibration/validation/test).

## Contents
- Resolutions: {", ".join(str(r) for r in sorted(resolutions))}
- Splits: {", ".join(split_names_sorted)}
- Total successful samples: {total_samples}
- {images_note}

## Structure

```
{repo_id}/
├── README.md
├── summary.json               # Overall generation summary
├── calibration/ (optional)    # calibrate.json + visualizations
├── {{resolution}}/            # 64, 128, 256, 512, 1024, 2048, 4096
│   ├── summary.json
│   ├── {{split}}/             # train, calibration, validation, test
│   │   ├── tensors/N045/E010/0000000000.npy
│   │   ├── metadata/N045/E010/0000000000.yaml
│   │   └── images/N045/E010/0000000000.png
│   └── ...
└── docs/ (optional notebooks)
```

## Split semantics
- **train**: training set
- **calibration**: hyperparameter / cost-model tuning
- **validation**: model selection
- **test**: held-out evaluation

## Resolution breakdown
{os.linesep.join(per_res_blocks)}

## Calibration (optional)
Global min/max elevation and per-tile statistics from ALOS AW3D30.
- Global Min Elevation: {calibrate_data.get("global_min", "N/A") if calibrate_data else "N/A"}
- Global Max Elevation: {calibrate_data.get("global_max", "N/A") if calibrate_data else "N/A"}
- Tiles: {len(calibrate_data.get("tile_stats", [])) if calibrate_data else "N/A"}

## Usage

```python
import numpy as np
from huggingface_hub import hf_hub_download

repo = "{repo_id}"
filename = "256/train/tensors/N045/E010/0000000000.npy"
tensor_path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
tensor = np.load(tensor_path)
```

## Source

Generated using [PASSAGE](https://github.com/YOUR_USERNAME/passage) from ALOS AW3D30 DSM tiles.

## License

MIT License
"""


# =============================================================================
# Prepare Functions - Create local export structure without uploading
# =============================================================================


def _execute_notebook_with_papermill(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
) -> bool:
    """Execute a notebook with papermill and save the output.

    Args:
        input_notebook: Path to input notebook.
        output_notebook: Path where executed notebook will be saved.
        parameters: Parameters to inject into the notebook.
        cwd: Working directory for notebook execution.

    Returns:
        True if execution succeeded, False otherwise.
    """
    if not PAPERMILL_AVAILABLE:
        logger.error(
            "papermill is not installed. Install it with:\n  uv add papermill\n  # or\n  pip install papermill"
        )
        return False

    try:
        import os

        logger.info(f"Executing notebook with parameters: {parameters}")

        # Use absolute paths for papermill
        input_abs = Path(input_notebook).resolve()
        output_abs = Path(output_notebook).resolve()

        old_cwd = os.getcwd()
        if cwd:
            os.chdir(cwd)
        try:
            pm.execute_notebook(
                str(input_abs),
                str(output_abs),
                parameters=parameters,
                kernel_name="python3",
            )
        finally:
            os.chdir(old_cwd)
        logger.info(f"✓ Notebook executed: {output_notebook.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to execute notebook {input_notebook.name}: {e}")
        return False


def _generate_main_readme(
    repo_prefix: str,
    resolutions: list[int],
    calibrate_data: dict | None,
    github_url: str = "https://github.com/YOUR_USERNAME/passage",
) -> str:
    """Generate main README for the export directory."""
    resolution_links = "\n".join(
        f"- [`{repo_prefix}-{res}`](./{res}/) - {res}×{res} resolution dataset" for res in sorted(resolutions)
    )

    calibration_info = ""
    if calibrate_data:
        calibration_info = f"""
## Calibration Data

Global elevation statistics from ALOS AW3D30 DSM tiles:
- **Global Min Elevation**: {calibrate_data.get("global_min", "N/A")} meters
- **Global Max Elevation**: {calibrate_data.get("global_max", "N/A")} meters
- **Total Tiles Analyzed**: {len(calibrate_data.get("tile_stats", []))} tiles

See [`calibration/`](./calibration/) for full calibration data and visualizations.
"""

    return f"""# PASSAGE Export

This directory (dataset root, e.g., outputs/Export) is ready to be uploaded to Hugging Face Hub.

## Structure

```
<dataset_root>/
├── README.md                    # This file
├── calibration/                 # Calibration data and visualizations
│   ├── README.md
│   ├── calibrate.json
│   └── *.png
├── docs/                        # Documentation notebooks (non-resolution-specific)
│   ├── README.md
│   └── notebooks/
│       ├── costmodel.ipynb
│       ├── metrics.ipynb
│       ├── obstacles.ipynb
│       ├── solve.ipynb
│       └── visualize.ipynb
├── 64/                          # Per-resolution data + executed dataset.ipynb
│   ├── README.md
│   ├── dataset.ipynb            # Executed with RESOLUTION=64
│   ├── summary.json
│   ├── train/|calibration/|validation/|test/
│   │   ├── tensors/<LAT>/<LON>/<index>.npy
│   │   ├── metadata/<LAT>/<LON>/<index>.yaml
│   │   ├── images/<LAT>/<LON>/<index>.png
│   │   └── manifest.parquet
│   └── ...
├── 256/
│   └── ...
└── 1024/
    └── ...
```

## Datasets by Resolution

{resolution_links}

{calibration_info}

Generated using [PASSAGE]({github_url}) from ALOS AW3D30 DSM tiles.

## License

MIT License
"""


def _generate_resolution_readme(
    resolution: int,
    calibrate_data: dict | None,
    summary_data: dict | None,
    repo_prefix: str,
) -> str:
    """Generate README for a resolution directory in prepared export."""
    num_samples = summary_data.get("num_samples", "N/A") if summary_data else "N/A"
    global_min = calibrate_data.get("global_min", "N/A") if calibrate_data else "N/A"
    global_max = calibrate_data.get("global_max", "N/A") if calibrate_data else "N/A"

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
size_categories:
  - 1K<n<10K
---

# PASSAGE Elevation Dataset - {resolution}×{resolution}

This dataset contains elevation data samples at {resolution}×{resolution} resolution for pathfinding tasks.

## Dataset Statistics

- **Resolution**: {resolution}×{resolution} pixels
- **Number of Samples**: {num_samples}
- **Global Min Elevation**: {global_min} meters
- **Global Max Elevation**: {global_max} meters

## Files

- `dataset.ipynb` - Executed analysis notebook with resolution-specific insights
- `summary.json` - Per-resolution summary
- Split folders: `train/`, `calibration/`, `validation/`, `test/`
    - `tensors/<LAT>/<LON>/<index>.npy`
    - `metadata/<LAT>/<LON>/<index>.yaml`
    - `images/<LAT>/<LON>/<index>.png` (if generated)
    - `manifest.parquet` (paths + identifiers for this split)

## Tensor Format

Each `.npy` tensor is a `({resolution}, {resolution}, 3)` float32 array:
- **Channel 0**: Normalized elevation (0-1 range using global min/max)
- **Channel 1**: Marker channel (-1 everywhere, 0 at start, 1 at end)
- **Channel 2**: Obstacles mask (1.0 for blocked, 0.0 for free)

## Usage

```python
import numpy as np
from pathlib import Path

# Load a tensor
tensor = np.load("data/tensors/N045/E010/0000000000.npy")
elevation = tensor[:, :, 0]
markers = tensor[:, :, 1]
obstacles = tensor[:, :, 2]
```

## Analysis

See `dataset.ipynb` for detailed analysis including:
- Path crossing heatmaps
- Elevation distributions
- Path length statistics
- Hausdorff distance analysis

## Related Datasets

- Calibration data: `{repo_prefix}-calibration`
- Documentation: `{repo_prefix}-docs`

## License

MIT License
"""


def _generate_docs_readme(repo_prefix: str, notebooks: list[str]) -> str:
    """Generate README for docs directory."""
    notebook_table = "\n".join(
        f"| [{nb}](notebooks/{nb}) | {nb.replace('.ipynb', '').replace('_', ' ').title()} analysis |"
        for nb in sorted(notebooks)
    )

    return f"""---
license: mit
tags:
  - passage
  - elevation
  - pathfinding
  - documentation
---

# PASSAGE Documentation Notebooks

This repository contains analysis notebooks for the PASSAGE elevation dataset.

## Notebooks

| Notebook | Description |
|----------|-------------|
{notebook_table}

## Note

These notebooks are **not** resolution-specific. For resolution-specific analysis
(dataset.ipynb executed with specific RESOLUTION parameter), see the individual
resolution datasets:

- `{repo_prefix}-64`
- `{repo_prefix}-256`
- `{repo_prefix}-1024`
- etc.

## Usage

Notebooks are rendered directly on Hugging Face. Click any notebook above to view.

To run locally:

```bash
# Clone the repository
git clone https://huggingface.co/datasets/{repo_prefix}-docs

# Install dependencies
pip install jupyter numpy pandas matplotlib passage

# Run Jupyter
jupyter notebook notebooks/
```

## License

MIT License
"""


def _generate_calibration_readme(calibrate_data: dict, repo_prefix: str) -> str:
    """Generate README for calibration directory."""
    return f"""---
license: mit
task_categories:
  - other
tags:
  - elevation
  - dsm
  - alos
  - calibration
  - passage
size_categories:
  - 10K<n<100K
---

# PASSAGE Calibration Data

Calibration data for the PASSAGE elevation dataset generator.

## Global Statistics

- **Global Min Elevation**: {calibrate_data.get("global_min", "N/A")} meters
- **Global Max Elevation**: {calibrate_data.get("global_max", "N/A")} meters
- **Total Tiles Analyzed**: {len(calibrate_data.get("tile_stats", []))} tiles

## Files

- `calibrate.json` - Full calibration data with per-tile statistics
- `max_elevation_map.png` - Visualization of max elevation per tile
- `ptp_elevation_map.png` - Peak-to-peak elevation range visualization

## Usage

```python
import json
from pathlib import Path

# Load calibration
with open("calibrate.json") as f:
    calibration = json.load(f)

global_min = calibration["global_min"]  # {calibrate_data.get("global_min", "N/A")}
global_max = calibration["global_max"]  # {calibrate_data.get("global_max", "N/A")}
tile_stats = calibration["tile_stats"]  # List of per-tile stats
```

## Related Datasets

- Documentation: `{repo_prefix}-docs`
- Resolution datasets: `{repo_prefix}-64`, `{repo_prefix}-256`, etc.

## License

MIT License
"""


def _generate_resolution_readme_for_prepare(
    resolution: int,
    calibrate_data: dict | None,
    summary_data: dict | None,
    repo_prefix: str,
) -> str:
    """Alias for resolution README generation used during prepare."""
    return _generate_resolution_readme(
        resolution=resolution,
        calibrate_data=calibrate_data,
        summary_data=summary_data,
        repo_prefix=repo_prefix,
    )


def prepare_export(
    export_dirpath: str | Path = "outputs/Export",
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    notebooks_dirpath: str | Path = "notebooks",
    resolutions: list[int] | None = None,
    repo_prefix: str = "passage",
    execute_notebooks: bool = True,
    max_samples_notebook: int | None = None,
) -> bool:
    """Prepare export artifacts in a clean outputs/Export directory.

    Creates under export_dirpath:
    - calibration/          (copied)
    - docs/notebooks/       (copied)
    - {resolution}/         (data referenced from dataset_dirpath) with:
        - Executed dataset.ipynb (via papermill)
        - summary.json copy
        - README.md
        - Split folders each with manifest.parquet
    - README.md at export root

    Args:
        export_dirpath: Directory where READMEs/notebooks/manifests are written (default: outputs/Export).
        dataset_dirpath: Path to source dataset directory (default: outputs/dataset).
        calibrate_dirpath: Path to outputs/calibrate directory.
        notebooks_dirpath: Path to notebooks directory.
        resolutions: List of resolutions to prepare. If None, uses all found.
        repo_prefix: Repository prefix for generated READMEs.
        execute_notebooks: Whether to execute dataset.ipynb with papermill.
        max_samples_notebook: Max samples for notebook execution (None = all).

    Returns:
        True if preparation succeeded, False otherwise.
    """
    export_path = Path(export_dirpath)
    dataset_path = Path(dataset_dirpath)

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False
    calibrate_path = Path(calibrate_dirpath)
    notebooks_path = Path(notebooks_dirpath)

    # Find available resolutions
    if resolutions is None:
        resolutions = []
        if dataset_path.exists():
            for d in dataset_path.iterdir():
                if d.is_dir() and d.name.isdigit():
                    resolutions.append(int(d.name))
        resolutions.sort()

    if not resolutions:
        logger.warning("No resolution directories found to prepare")
        return False

    logger.info(f"Preparing export for resolutions: {resolutions}")

    # Create export directory fresh
    export_path.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    calibrate_data = None
    calibrate_json = calibrate_path / "calibrate.json"
    if calibrate_json.exists():
        with open(calibrate_json) as f:
            calibrate_data = json.load(f)

    # 1. Prepare calibration directory (in-place)
    calibration_export = export_path / "calibration"
    calibration_export.mkdir(exist_ok=True)

    if calibrate_path.exists():
        logger.info("Preparing calibration data...")
        for src_file in calibrate_path.iterdir():
            if src_file.is_file() and not src_file.name.endswith(".bak"):
                shutil.copy2(src_file, calibration_export / src_file.name)

        # Generate calibration README
        if calibrate_data:
            readme = _generate_calibration_readme(calibrate_data, repo_prefix)
            (calibration_export / "README.md").write_text(readme)
        logger.info(f"✓ Calibration prepared: {calibration_export}")

    # 2. Prepare docs directory (non-resolution notebooks)
    docs_export = export_path / "docs"
    docs_export.mkdir(exist_ok=True)
    docs_notebooks = docs_export / "notebooks"
    docs_notebooks.mkdir(exist_ok=True)

    # Copy all notebooks except dataset.ipynb to docs
    doc_notebook_names = []
    if notebooks_path.exists():
        logger.info("Preparing documentation notebooks...")
        for nb in notebooks_path.glob("*.ipynb"):
            if nb.name != "dataset.ipynb":
                shutil.copy2(nb, docs_notebooks / nb.name)
                doc_notebook_names.append(nb.name)
                logger.info(f"  Copied {nb.name}")

        # Generate docs README
        readme = _generate_docs_readme(repo_prefix, doc_notebook_names)
        (docs_export / "README.md").write_text(readme)
        logger.info(f"✓ Documentation prepared: {docs_export}")

    split_names = ["train", "calibration", "validation", "test"]

    # 3. Prepare per-resolution directories (operate in-place)
    dataset_notebook = notebooks_path / "dataset.ipynb"

    for res in resolutions:
        res_export = export_path / str(res)
        res_export.mkdir(exist_ok=True)

        res_data = dataset_path / str(res)
        if not res_data.exists():
            logger.warning(f"Resolution {res} data not found at {res_data}, skipping")
            continue

        logger.info(f"Preparing resolution {res}...")

        # Load summary data
        summary_data = None
        summary_json = res_data / "summary.json"
        if summary_json.exists():
            with open(summary_json) as f:
                summary_data = json.load(f)
            shutil.copy2(summary_json, res_export / "summary.json")

        # Execute dataset.ipynb with papermill (write alongside data)
        if execute_notebooks and dataset_notebook.exists():
            output_notebook = res_export / "dataset.ipynb"
            # Default MAX_SAMPLES to 0 during prepare to avoid full dataset processing
            # (the notebook will run quickly and not iterate over all samples)
            parameters = {
                "RESOLUTION": res,
                "MAX_SAMPLES": 0 if max_samples_notebook is None else max_samples_notebook,
                "SAVE_WORLD_COVER": False,  # Don't save extra files
            }

            if PAPERMILL_AVAILABLE:
                success = _execute_notebook_with_papermill(
                    dataset_notebook,
                    output_notebook,
                    parameters,
                    cwd=notebooks_path,  # Execute from notebooks/ directory
                )
                if not success:
                    # Copy notebook without execution as fallback
                    shutil.copy2(dataset_notebook, output_notebook)
                    logger.warning("  Copied notebook without execution (papermill failed)")
            else:
                # Copy without execution
                shutil.copy2(dataset_notebook, output_notebook)
                logger.warning("  Copied notebook without execution (papermill not available)")
        elif dataset_notebook.exists():
            # Copy without execution
            shutil.copy2(dataset_notebook, res_export / "dataset.ipynb")
            logger.info("  Copied dataset.ipynb (execution disabled)")

        # Generate resolution README
        readme = _generate_resolution_readme_for_prepare(res, calibrate_data, summary_data, repo_prefix)
        (res_export / "README.md").write_text(readme)
        logger.info(f"✓ Resolution {res} prepared: {res_export}")

        # Generate per-split parquet manifests in export directory
        for split in split_names:
            split_src = res_data / split
            if not split_src.exists():
                continue

            tensors_root = split_src / "tensors"
            metadata_root = split_src / "metadata"
            images_root = split_src / "images"

            rows = []
            if metadata_root.exists():
                for meta_file in metadata_root.rglob("*.yaml"):
                    try:
                        idx = int(meta_file.stem)
                    except ValueError:
                        continue
                    parts = meta_file.parts
                    if len(parts) < 3:
                        continue
                    lat_dir = meta_file.parent.parent.name
                    lon_dir = meta_file.parent.name
                    tensor_path = tensors_root / lat_dir / lon_dir / f"{idx:010d}.npy"
                    image_path = images_root / lat_dir / lon_dir / f"{idx:010d}.png"
                    rows.append(
                        {
                            "index": idx,
                            "resolution": res,
                            "split": split,
                            "lat": lat_dir,
                            "lon": lon_dir,
                            "tensor_path": str(tensor_path.relative_to(dataset_path)),
                            "metadata_path": str(meta_file.relative_to(dataset_path)),
                            "image_path": str(image_path.relative_to(dataset_path)) if image_path.exists() else None,
                        }
                    )

            if rows:
                df = pd.DataFrame(rows)
                split_export = res_export / split
                split_export.mkdir(parents=True, exist_ok=True)
                manifest_path = split_export / "manifest.parquet"
                _try_write_parquet(df, manifest_path)

    # 4. Generate main README in-place
    main_readme = _generate_main_readme(repo_prefix, resolutions, calibrate_data)
    (export_path / "README.md").write_text(main_readme)

    logger.info(f"\n✓ Export prepared in-place at: {export_path}")
    logger.info(f"  Resolutions: {resolutions}")
    logger.info(f"  Calibration: {calibration_export}")
    logger.info(f"  Documentation: {docs_export}")
    logger.info("\nNext steps:")
    logger.info("  1. Review the prepared dataset directory")
    logger.info("  2. Upload with: uv run passage export <repo_prefix>")

    return True


def _create_or_update_repo(
    repo_id: str,
    private: bool = False,
) -> str:
    """Create or get existing HF dataset repository."""
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        logger.info(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
        return repo_id
    except Exception as e:
        logger.error(f"Failed to create/access repository {repo_id}: {e}")
        raise


def export_calibration(
    calibrate_dirpath: str | Path,
    repo_id: str,
    private: bool = False,
) -> bool:
    """Export calibration data to Hugging Face Hub.

    Args:
        calibrate_dirpath: Path to outputs/calibrate directory.
        repo_id: Hugging Face repository ID (e.g., 'username/passage-calibration').
        private: Whether to make the repository private.

    Returns:
        True if export succeeded, False otherwise.
    """
    if not _check_hf_available() or not _check_hf_login():
        return False

    calibrate_path = Path(calibrate_dirpath)
    calibrate_json = calibrate_path / "calibrate.json"

    if not calibrate_json.exists():
        logger.error(f"Calibration file not found: {calibrate_json}")
        return False

    # Load calibration data for README
    with open(calibrate_json) as f:
        calibrate_data = json.load(f)

    try:
        _create_or_update_repo(repo_id, private=private)

        # Create temporary directory with README
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Copy calibration files
            for src_file in calibrate_path.iterdir():
                if src_file.is_file() and not src_file.name.endswith(".bak"):
                    shutil.copy2(src_file, tmpdir / src_file.name)

            # Generate README
            readme_content = _generate_dataset_card(
                repo_id=repo_id,
                resolution=None,
                calibrate_data=calibrate_data,
                summary_data=None,
                is_calibration=True,
            )
            (tmpdir / "README.md").write_text(readme_content)

            # Upload
            upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload PASSAGE calibration data",
            )

        logger.info(f"✓ Calibration data exported to: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to export calibration data: {e}")
        return False


def export_resolution(
    dataset_dirpath: str | Path,
    resolution: int,
    repo_id: str,
    calibrate_dirpath: str | Path | None = None,
    private: bool = False,
    include_images: bool = True,
) -> bool:
    """Legacy function (unused in single-repo mode)."""
    logger.warning("export_resolution is unused in single-repo export mode.")
    return False


def _try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    """Write dataframe to parquet if engine available; returns True on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return True
    except ImportError:
        logger.warning("pyarrow or fastparquet not installed; skipping parquet write for %s", path)
        return False
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to write parquet {path}: {e}")
        return False


def export_notebooks(
    notebooks_dirpath: str | Path,
    repo_id: str,
    private: bool = False,
) -> bool:
    """Export notebooks as documentation to a Hugging Face Space or dataset.

    This function exports Jupyter notebooks to Hugging Face. Notebooks can be
    used as documentation and are rendered directly on the Hub.

    Args:
        notebooks_dirpath: Path to notebooks directory.
        repo_id: Hugging Face repository ID (e.g., 'username/passage-docs').
        private: Whether to make the repository private.

    Returns:
        True if export succeeded, False otherwise.
    """
    if not _check_hf_available() or not _check_hf_login():
        return False

    notebooks_path = Path(notebooks_dirpath)

    if not notebooks_path.exists():
        logger.error(f"Notebooks directory not found: {notebooks_path}")
        return False

    notebooks = list(notebooks_path.glob("*.ipynb"))
    if not notebooks:
        logger.warning(f"No notebooks found in {notebooks_path}")
        return False

    try:
        _create_or_update_repo(repo_id, private=private)

        # Create temporary directory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create notebooks subdirectory
            (tmpdir / "notebooks").mkdir()

            # Copy notebooks
            for nb in notebooks:
                shutil.copy2(nb, tmpdir / "notebooks" / nb.name)

            # Generate README that links to notebooks
            readme_content = """---
license: mit
tags:
  - passage
  - elevation
  - pathfinding
  - documentation
---

# PASSAGE Documentation Notebooks

This repository contains Jupyter notebooks documenting the PASSAGE elevation dataset.

## Notebooks

| Notebook | Description |
|----------|-------------|
"""
            for nb in sorted(notebooks):
                name = nb.stem.replace("_", " ").title()
                readme_content += f"| [{nb.name}](notebooks/{nb.name}) | {name} |\n"

            readme_content += """
## Viewing Notebooks

Notebooks are rendered directly on Hugging Face. Click on any notebook above to view it.

You can also download and run them locally:

```bash
# Clone the repository
git clone https://huggingface.co/datasets/{repo_id}
cd {repo_name}

# Install dependencies
pip install jupyter numpy pandas matplotlib

# Run Jupyter
jupyter notebook notebooks/
```

## Source

These notebooks are part of [PASSAGE](https://github.com/YOUR_USERNAME/passage).
""".format(repo_id=repo_id, repo_name=repo_id.split("/")[-1])

            (tmpdir / "README.md").write_text(readme_content)

            # Upload
            logger.info(f"Uploading notebooks to {repo_id}...")
            upload_folder(
                folder_path=str(tmpdir),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload PASSAGE documentation notebooks",
            )

        logger.info(f"✓ Notebooks exported to: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to export notebooks: {e}")
        return False


def run_export(
    repo_prefix: str,
    dataset_dirpath: str = "outputs/dataset",
    calibrate_dirpath: str = "outputs/calibrate",
    notebooks_dirpath: str = "notebooks",
    resolutions: list[int] | None = None,
    include_calibration: bool = True,
    include_notebooks: bool = False,
    include_images: bool = True,
    private: bool = False,
    write_manifests: bool = True,
) -> None:
    """Export PASSAGE datasets to a single Hugging Face dataset with split-aware layout.

    Layout in the repo:
    {resolution}/{split}/{tensors,metadata,images}/{LAT}/{LON}/{index}.ext
    """

    if not _check_hf_available() or not _check_hf_login():
        return

    dataset_path = Path(dataset_dirpath)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return
    calibrate_path = Path(calibrate_dirpath)
    notebooks_path = Path(notebooks_dirpath)

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return

    # Find available resolutions
    if resolutions is None:
        resolutions = []
        for d in dataset_path.iterdir():
            if d.is_dir() and d.name.isdigit():
                resolutions.append(int(d.name))
        resolutions.sort()

    if not resolutions:
        logger.warning("No resolution directories found to export")
        return

    split_names = ["train", "calibration", "validation", "test"]
    repo_id = repo_prefix

    calibrate_data = None
    if include_calibration:
        calibrate_json = calibrate_path / "calibrate.json"
        if calibrate_json.exists():
            with open(calibrate_json) as f:
                calibrate_data = json.load(f)

    summaries: dict[int, dict] = {}
    for res in resolutions:
        summary_json = dataset_path / str(res) / "summary.json"
        if summary_json.exists():
            try:
                with open(summary_json) as f:
                    summaries[res] = json.load(f)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to read summary for {res}: {e}")

    overall_summary_path = dataset_path / "summary.json"

    try:
        _create_or_update_repo(repo_id, private=private)
    except Exception:
        return

    # Upload README + summaries (+ optional calibration/notebooks)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readme_content = _generate_multi_dataset_card(
                repo_id=repo_id,
                resolutions=resolutions,
                split_names=split_names,
                summaries=summaries,
                calibrate_data=calibrate_data,
                include_images=include_images,
            )
            (tmp / "README.md").write_text(readme_content)

            if overall_summary_path.exists():
                shutil.copy2(overall_summary_path, tmp / "summary.json")

            for res in resolutions:
                res_summary = dataset_path / str(res) / "summary.json"
                if res_summary.exists():
                    res_tmp = tmp / str(res)
                    res_tmp.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(res_summary, res_tmp / "summary.json")

            if include_calibration and calibrate_data:
                calib_tmp = tmp / "calibration"
                calib_tmp.mkdir(parents=True, exist_ok=True)
                for src_file in calibrate_path.iterdir():
                    if src_file.is_file() and not src_file.name.endswith(".bak"):
                        shutil.copy2(src_file, calib_tmp / src_file.name)

            if include_notebooks and notebooks_path.exists():
                docs_tmp = tmp / "docs"
                docs_tmp.mkdir(parents=True, exist_ok=True)
                docs_notebooks = docs_tmp / "notebooks"
                docs_notebooks.mkdir(parents=True, exist_ok=True)
                for nb in notebooks_path.glob("*.ipynb"):
                    shutil.copy2(nb, docs_notebooks / nb.name)

            upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload PASSAGE dataset metadata and summaries",
            )
            logger.info("✓ Uploaded README, summaries, and optional calibration/notebooks")
    except Exception as e:
        logger.error(f"Failed to upload metadata to {repo_id}: {e}")
        return

    # Optional: regenerate parquet manifests in-place before upload
    manifest_paths: list[tuple[Path, str]] = []
    if write_manifests:
        for res in resolutions:
            res_root = dataset_path / str(res)
            for split in split_names:
                split_root = res_root / split
                if not split_root.exists():
                    continue
                tensors_root = split_root / "tensors"
                metadata_root = split_root / "metadata"
                images_root = split_root / "images"

                rows = []
                if metadata_root.exists():
                    for meta_file in metadata_root.rglob("*.yaml"):
                        try:
                            idx = int(meta_file.stem)
                        except ValueError:
                            continue
                        parts = meta_file.parts
                        # .../metadata/<LAT>/<LON>/<file>
                        if len(parts) < 3:
                            continue
                        lat_dir = meta_file.parent.parent.name
                        lon_dir = meta_file.parent.name
                        tensor_path = tensors_root / lat_dir / lon_dir / f"{idx:010d}.npy"
                        image_path = images_root / lat_dir / lon_dir / f"{idx:010d}.png"
                        rows.append(
                            {
                                "index": idx,
                                "resolution": res,
                                "split": split,
                                "lat": lat_dir,
                                "lon": lon_dir,
                                "tensor_path": str(tensor_path.relative_to(dataset_path)),
                                "metadata_path": str(meta_file.relative_to(dataset_path)),
                                "image_path": str(image_path.relative_to(dataset_path))
                                if include_images and image_path.exists()
                                else None,
                            }
                        )

                if rows:
                    df = pd.DataFrame(rows)
                    manifest_path = split_root / "manifest.parquet"
                    if _try_write_parquet(df, manifest_path):
                        rel = manifest_path.relative_to(dataset_path)
                        manifest_paths.append((manifest_path, str(rel).replace(os.sep, "/")))

    # Chunked data upload by LAT/LON
    failures = 0
    successes = 0
    for res in resolutions:
        res_root = dataset_path / str(res)
        if not res_root.exists():
            logger.warning(f"Resolution {res} not found, skipping")
            continue

        for split in split_names:
            split_root = res_root / split
            if not split_root.exists():
                logger.debug(f"Split {split} missing for resolution {res}, skipping")
                continue

            for kind in ["tensors", "metadata", "images"]:
                if kind == "images" and not include_images:
                    continue
                kind_root = split_root / kind
                if not kind_root.exists():
                    continue

                for lat_dir in sorted([p for p in kind_root.iterdir() if p.is_dir()]):
                    for lon_dir in sorted([p for p in lat_dir.iterdir() if p.is_dir()]):
                        rel_path = f"{res}/{split}/{kind}/{lat_dir.name}/{lon_dir.name}"
                        try:
                            logger.info(f"Uploading chunk {rel_path}...")
                            upload_folder(
                                folder_path=str(lon_dir),
                                repo_id=repo_id,
                                repo_type="dataset",
                                path_in_repo=rel_path,
                                commit_message=f"Upload {rel_path}",
                            )
                            successes += 1
                        except Exception as e:  # pragma: no cover
                            failures += 1
                            logger.error(f"Failed to upload chunk {rel_path}: {e}")

    # Upload manifests (single files)
    if manifest_paths:
        api = HfApi()
        for manifest_path, rel in manifest_paths:
            try:
                logger.info(f"Uploading manifest {rel}...")
                api.upload_file(
                    path_or_fileobj=str(manifest_path),
                    path_in_repo=rel,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload manifest {rel}",
                )
                successes += 1
            except Exception as e:  # pragma: no cover
                failures += 1
                logger.error(f"Failed to upload manifest {rel}: {e}")

    logger.info(f"Export completed: {successes} chunks uploaded, {failures} failed")
