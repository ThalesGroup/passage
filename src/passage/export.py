"""Prepare and push PASSAGE datasets directly to Hugging Face.

This refactor drops intermediate parquet creation and writes datasets directly
to the Hub using `datasets.Dataset.push_to_hub`, with zstd-compressed shards
controlled by the `export.chunk_size.<resolution>` configuration (rows per shard)
and a `max_shard_size` target of ~10 GB.
"""

import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

import httpcore
import httpx
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Features, Image, List, Sequence, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger
from tqdm import tqdm

try:  # Optional papermill fallback
    import papermill as pm

    PAPERMILL_AVAILABLE = True
except Exception:  # pragma: no cover - papermill optional
    PAPERMILL_AVAILABLE = False


def parse_path_csv(csv_path: Path) -> list[list[int]]:
    """Parse a path .csv file with rows "x,y" (no header) and return [[row,col], ...]."""

    out: list[list[int]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            x = int(row[0].strip())
            y = int(row[1].strip())
            out.append([x, y])
    return out


def _process_meta_for_prepare(resolution: int, sample: bool, metadata_filepath: str) -> dict:
    """Process a single metadata file into a serializable dict for HF dataset rows."""

    metadata_filepath = Path(metadata_filepath)
    split_dirpath = Path(metadata_filepath.parent.parent.parent.parent)
    lat_dir = metadata_filepath.parent.parent.name
    lon_dir = metadata_filepath.parent.name
    stem = metadata_filepath.stem

    tensor_filepath = split_dirpath / "tensors" / lat_dir / lon_dir / f"{stem}.npy.zst"
    image_filepath = split_dirpath / "images" / lat_dir / lon_dir / f"{stem}.png"
    path_free_filepath = split_dirpath / "paths" / "free" / lat_dir / lon_dir / f"{stem}.csv"
    path_obstacles_filepath = split_dirpath / "paths" / "obstacles" / lat_dir / lon_dir / f"{stem}.csv"

    metadata = yaml.safe_load(metadata_filepath.read_text())

    # image_bytes = None
    # if image_filepath.exists():
    #     with PilImage.open(image_filepath) as im:
    #         buf = io.BytesIO()
    #         im.save(buf, format="PNG")
    #         image_bytes = buf.getvalue()

    # return {
    #     # "image": image_bytes,  # TODO: replace by .read_bytes()?
    #     "image": image_filepath.read_bytes(),
    #     "id": str(stem),
    #     # "tensor": tensor_filepath.read_bytes(),
    #     "tensor": load_tensor(tensor_filepath, shape=(resolution, resolution, 3), dtype="float32"),
    #     "path_free": parse_path_csv(path_free_filepath),
    #     "path_obstacles": parse_path_csv(path_obstacles_filepath),
    #     "metadata": metadata,
    # }

    sample_dict = {
        "id": str(stem),
        # "tensor": load_tensor(tensor_filepath, shape=(resolution, resolution, 3), dtype="float32")
        "tensor": tensor_filepath.read_bytes(),
        # "metadata": json.dumps(metadata),
        "metadata": metadata,
        "path_free": parse_path_csv(path_free_filepath),
        "path_obstacles": parse_path_csv(path_obstacles_filepath),
    }
    if sample:
        sample_dict["image"] = {"bytes": image_filepath.read_bytes()}
    return sample_dict


def _features(sample: bool) -> dict:
    features_dict = {
        "id": Value("string"),
        "path_free": Sequence(Sequence(Value("int32"))),
        "path_obstacles": Sequence(Sequence(Value("int32"))),
        # "metadata": Value("string"),
        "metadata": {
            "index": Value("int32"),
            "resolution": Value("int32"),
            "split": Value("string"),
            "target_elevation": Value("float32"),
            "calibration": {
                "min": Value("float32"),
                "max": Value("float32"),
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
            "paths": {
                "without_obstacles": {
                    "length": Value("float32"),
                    "pixels": List(
                        {
                            "i": Value("int32"),
                            "j": Value("int32"),
                            "elevation": Value("float32"),
                        }
                    ),
                    "crosses_obstacles": Value("bool"),
                    "cost": Value("float32"),
                },
                "with_obstacles": {
                    "length": Value("float32"),
                    "pixels": List(
                        {
                            "i": Value("int32"),
                            "j": Value("int32"),
                            "elevation": Value("float32"),
                        }
                    ),
                    "crosses_obstacles": Value("bool"),
                    "cost": Value("float32"),
                    "cost_diff_percent": Value("float32"),
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
            "solver": {
                "backend": Value("string"),
                "name": Value("string"),
                "diagonal_movement": Value("string"),
                "weight": Value("float32"),
                "heuristic": Value("string"),
                "cost_model": Value("string"),
            },
            "timing": {
                "tile_loading_seconds": Value("float32"),
                "marker_search_seconds": Value("float32"),
                "marker_find_path_without_seconds": Value("float32"),
                "marker_find_path_with_seconds": Value("float32"),
                "graph_prebuild_seconds": Value("float32"),
                "obstacle_generation_seconds": Value("float32"),
                "cost_compute_seconds": Value("float32"),
                "tensor_save_seconds": Value("float32"),
                "metadata_write_seconds": Value("float32"),
                "image_conversion_seconds": Value("float32"),
                "image_write_seconds": Value("float32"),
            },
            "timestamp": Value("timestamp[us]"),
        },
    }
    # features_dict["tensor"] = Sequence(Sequence(Sequence(Value("float32"))))
    features_dict["tensor"] = Value("binary")
    if sample:
        features_dict["image"] = Image()
        # features_dict["tensor"] = Sequence(Sequence(Sequence(Value("float32"))))
    else:
        # features_dict["tensor"] = Value("binary")
        pass
    return Features(features_dict)


FEATURES = Features(
    {
        "id": Value("string"),
        "image": Image(),
        "tensor": Value("binary"),
        # "tensor": Sequence(Sequence(Sequence(Value("float32")))),
        # "metadata": Value("string"),
        "metadata": {
            "index": Value("int32"),
            "resolution": Value("int32"),
            "split": Value("string"),
            "target_elevation": Value("float32"),
            "calibration": {
                "min": Value("float32"),
                "max": Value("float32"),
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
            "paths": {
                "without_obstacles": {
                    "length": Value("float32"),
                    "pixels": List(
                        {
                            "i": Value("int32"),
                            "j": Value("int32"),
                            "elevation": Value("float32"),
                        }
                    ),
                    "crosses_obstacles": Value("bool"),
                    "cost": Value("float32"),
                },
                "with_obstacles": {
                    "length": Value("float32"),
                    "pixels": List(
                        {
                            "i": Value("int32"),
                            "j": Value("int32"),
                            "elevation": Value("float32"),
                        }
                    ),
                    "crosses_obstacles": Value("bool"),
                    "cost": Value("float32"),
                    "cost_diff_percent": Value("float32"),
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
            "solver": {
                "backend": Value("string"),
                "name": Value("string"),
                "diagonal_movement": Value("string"),
                "weight": Value("float32"),
                "heuristic": Value("string"),
                "cost_model": Value("string"),
            },
            "timing": {
                "tile_loading_seconds": Value("float32"),
                "marker_search_seconds": Value("float32"),
                "marker_find_path_without_seconds": Value("float32"),
                "marker_find_path_with_seconds": Value("float32"),
                "graph_prebuild_seconds": Value("float32"),
                "obstacle_generation_seconds": Value("float32"),
                "cost_compute_seconds": Value("float32"),
                "tensor_save_seconds": Value("float32"),
                "metadata_write_seconds": Value("float32"),
                "image_conversion_seconds": Value("float32"),
                "image_write_seconds": Value("float32"),
            },
            "timestamp": Value("timestamp[us]"),
        },
        "path_free": Sequence(Sequence(Value("int32"))),
        "path_obstacles": Sequence(Sequence(Value("int32"))),
    }
)


def _execute_notebook_with_papermill(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict,
    cwd: Path | None = None,
) -> bool:
    """Execute a notebook with papermill using uv, with fallback to papermill API."""

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
) -> bool:
    attempt = 0
    notebook_name = input_notebook.name
    while True:
        attempt += 1
        resolution_info = f" (resolution={parameters.get('RESOLUTION')})" if "RESOLUTION" in parameters else ""
        logger.info(f"Executing notebook {notebook_name} (attempt {attempt}){resolution_info}")
        try:
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
    return f"{resolution}x{resolution}" + ("_sample" if sample else "")


def _generate_huggingface_yaml_header(
    repo_id: str, resolutions: list[int], split_names: list[str], default_config_name: str
) -> str:
    _ = repo_id  # retained for potential future metadata injection

    dataset_info_configs: list[str] = []
    for res in sorted(resolutions):
        for sample in [True, False]:
            dataset_info_configs.append(f"  - config_name: {_config_name(res, sample)}")

    dataset_config = {
        "dataset_info": [],
        "configs": [],
        "language": ["en"],
        "license": ["mit"],
        "task_categories": ["image-segmentation", "other"],
        "tags": ["elevation", "pathfinding", "dsm", "alos", "passage", "terrain", "obstacles", "A*"],
        "pretty_name": "PASSAGE",
        "size_categories": ["100K<n<1M"],
    }

    for res in resolutions:
        for sample in [True, False]:
            config_name = _config_name(res, sample)
            dataset_info_config = {
                "config_name": config_name,
                # "dataset_size": _dir_size_bytes(Path("outputs") / "export" / config_name, sample),
                # "download_size": _dir_size_bytes(Path("outputs") / "export" / config_name, sample),
                "features": [
                    {"name": "id", "dtype": "string"},
                    # {"name": "tensor", "list": {"list": {"list": "float32"}}},
                    {"name": "tensor", "dtype": "binary"},
                    {"name": "path_free", "list": {"list": "int32"}},
                    {"name": "path_obstacles", "list": {"list": "int32"}},
                    # {"name": "metadata", "dtype": "string"},
                    {
                        "name": "metadata",
                        "struct": [
                            {"name": "index", "dtype": "int32"},
                            {"name": "resolution", "dtype": "int32"},
                            {"name": "split", "dtype": "string"},
                            {"name": "target_elevation", "dtype": "float32"},
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
                                "name": "paths",
                                "struct": [
                                    {
                                        "name": "without_obstacles",
                                        "struct": [
                                            {"name": "length", "dtype": "float32"},
                                            {
                                                "name": "pixels",
                                                "list": {
                                                    "struct": [
                                                        {"name": "i", "dtype": "int32"},
                                                        {"name": "j", "dtype": "int32"},
                                                        {"name": "elevation", "dtype": "float32"},
                                                    ]
                                                },
                                            },
                                            {"name": "crosses_obstacles", "dtype": "bool"},
                                            {"name": "cost", "dtype": "float32"},
                                        ],
                                    },
                                    {
                                        "name": "with_obstacles",
                                        "struct": [
                                            {"name": "length", "dtype": "float32"},
                                            {
                                                "name": "pixels",
                                                "list": {
                                                    "struct": [
                                                        {"name": "i", "dtype": "int32"},
                                                        {"name": "j", "dtype": "int32"},
                                                        {"name": "elevation", "dtype": "float32"},
                                                    ]
                                                },
                                            },
                                            {"name": "crosses_obstacles", "dtype": "bool"},
                                            {"name": "cost", "dtype": "float32"},
                                            {"name": "cost_diff_percent", "dtype": "float32"},
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
                                "name": "solver",
                                "struct": [
                                    {"name": "backend", "dtype": "string"},
                                    {"name": "name", "dtype": "string"},
                                    {"name": "diagonal_movement", "dtype": "string"},
                                    {"name": "weight", "dtype": "float32"},
                                    {"name": "heuristic", "dtype": "string"},
                                    {"name": "cost_model", "dtype": "string"},
                                ],
                            },
                            {
                                "name": "timing",
                                "struct": [
                                    {"name": "tile_loading_seconds", "dtype": "float32"},
                                    {"name": "marker_search_seconds", "dtype": "float32"},
                                    {"name": "marker_find_path_without_seconds", "dtype": "float32"},
                                    {"name": "marker_find_path_with_seconds", "dtype": "float32"},
                                    {"name": "graph_prebuild_seconds", "dtype": "float32"},
                                    {"name": "obstacle_generation_seconds", "dtype": "float32"},
                                    {"name": "cost_compute_seconds", "dtype": "float32"},
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
                # "splits": [],
            }
            if sample:
                dataset_info_config["features"].insert(1, {"name": "image", "dtype": "image"})
            configs_config = {
                "config_name": config_name,
                "data_files": [],
            }
            if config_name == default_config_name:
                configs_config["default"] = True
            # num_samples = 0
            for split in split_names:
                # num_samples_split = _dir_num_files(Path("outputs") / "dataset" / config_name / split / "tensors")
                # dataset_info_config["splits"].append(
                #     {
                #         "name": split,
                #         # "num_bytes": _dir_size_bytes(Path("outputs") / "export" / config_name / split, sample),
                #         "num_examples": num_samples_split,
                #     }
                # )
                # num_samples += num_samples_split
                configs_config["data_files"].append(
                    {
                        "split": split,
                        "path": f"{config_name}/{split}/*",
                    }
                )
            # dataset_info_config["num_examples"] = num_samples
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


def _load_hf_credentials(
    env_path: Path | None = None,
    secret_path: Path | None = None,
) -> tuple[str, str, str]:
    project_root = Path(__file__).parent.parent.parent

    if env_path is None:
        env_path = project_root / ".env"
    if secret_path is None:
        secret_path = project_root / ".env.secret"

    if env_path.exists():
        load_dotenv(env_path)
    if secret_path.exists():
        load_dotenv(secret_path)

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


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    hf_rows = []
    for r in rows:
        hf_rows.append(
            {
                "image": {"bytes": r["image"]} if r.get("image") is not None else None,
                "id": r["id"],
                "tensor": r["tensor"],
                "metadata": json.dumps(r.get("metadata", {})),
                "path_free": r.get("path_free", []),
                "path_obstacles": r.get("path_obstacles", []),
            }
        )
    return Dataset.from_list(hf_rows, features=FEATURES)


def _load_rows_for_split(split_dir: Path, resolution: int) -> list[dict]:
    metadata_root = split_dir / "metadata"
    metadata_filepaths = sorted(p for p in metadata_root.rglob("*.yaml") if p.stem.isdigit())
    rows: list[dict] = []
    for p in metadata_filepaths:
        rows.append(_process_meta_for_prepare(str(p), resolution))
    return rows


def _push_resolution(
    api: HfApi,
    export_dirpath: Path,
    token: str,
    repo_id: str,
    resolution: int,
    split_datasets: dict[str, Dataset],
    private: bool,
    chunk_size: int,
) -> None:
    config_name = f"{resolution}x{resolution}"

    max_retries = int(os.getenv("HF_UPLOAD_RETRIES", "3"))
    base_delay = float(os.getenv("HF_UPLOAD_RETRY_DELAY", "5"))
    max_delay = float(os.getenv("HF_UPLOAD_MAX_DELAY", "60"))

    for split, ds in split_datasets.items():
        attempt = 0
        while True:
            attempt += 1
            try:
                # TODO: generate parquet compressed and chunked files then upload using api.upload_folder
                # TODO: rewrite the README.md hugging face header
                ds.push_to_hub(
                    repo_id,
                    config_name=config_name,
                    split=split,
                    private=private,
                    # num_shards=len(ds) // chunk_size + 1,
                    max_shard_size="10GB",
                    token=token,
                    set_default=resolution == 1024 and split == "train",
                    data_dir=config_name + "/" + split,
                    num_proc=min(max(os.cpu_count() - 1, 1), len(ds)),
                )
                logger.info(f"✓ Pushed {config_name}/{split} (attempt {attempt})")
                break
            except (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.HTTPError) as exc:
                if attempt >= max_retries:
                    logger.error(f"Upload failed for {config_name}/{split} after {attempt} attempts: {exc}")
                    raise
                sleep_for = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Push to hub timed out for {config_name}/{split} (attempt {attempt}/{max_retries}); retrying in {sleep_for}s: {exc}"
                )
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                raise


def write_shard(info):
    i, num, dataset, export_dirpath = info
    shard = dataset.shard(num_shards=num, index=i)
    shard.to_parquet(f"{export_dirpath}/data-{i:05d}-of-{num:05d}.parquet", compression="zstd")


def _create_parquets(
    export_dirpath: Path,
    resolution: int,
    sample: bool,
    split_datasets: dict[str, Dataset],
    chunk_size: int,
) -> None:
    config_name = _config_name(resolution, sample)
    for split, ds in split_datasets.items():
        export_path = export_dirpath / config_name / split
        export_path.mkdir(parents=True, exist_ok=True)
        num_chunks = len(ds) // chunk_size + (1 if len(ds) % chunk_size != 0 else 0)
        with mp.Pool(processes=min(max(os.cpu_count() - 1, 1), num_chunks)) as pool:
            list(
                tqdm(
                    pool.imap(
                        write_shard,
                        [(i, num_chunks, ds, str(export_path.resolve())) for i in range(num_chunks)],
                    ),
                    total=num_chunks,
                    desc=f"Exporting {config_name}/{split}",
                    unit="chunk",
                )
            )


def _upload_readme(api: HfApi, repo_id: str, readme_text: str) -> None:
    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def _upload_notebook(api: HfApi, repo_id: str, notebook_path: Path, resolution: int) -> None:
    config_name = f"{resolution}x{resolution}"
    api.upload_file(
        path_or_fileobj=str(notebook_path.resolve()),
        path_in_repo=config_name + "/dataset.ipynb",
        repo_id=repo_id,
        repo_type="dataset",
    )


def export_and_upload(
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    notebooks_dirpath: str | Path = "notebooks",
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
) -> bool:
    dataset_path = Path(dataset_dirpath).resolve()
    calibrate_path = Path(calibrate_dirpath).resolve()
    notebooks_path = Path(notebooks_dirpath).resolve()
    dataset_notebook = notebooks_path / "_dataset.ipynb"
    export_dirpath = calibrate_path.parent / "export"
    export_dirpath.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False

    config_file = notebooks_path.parent / "config" / "config.yaml"
    cfg = yaml.safe_load(config_file.read_text())
    chunk_sizes = cfg["export"]["chunk_size"]

    if resolutions is None:
        resolutions = [int(d.name) for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]

    split_names = ["train", "calibration", "validation", "test"]

    # if skip_upload:
    #     logger.info("Skipping upload as requested; notebooks (if any) executed")
    #     return True

    # TODO: extract getting username and repo_name from env to a separate function
    username, repo_name, token = _load_hf_credentials(
        env_path=Path(env_path) if env_path else None,
        secret_path=Path(secret_path) if secret_path else None,
    )
    if repo_id is None:
        repo_id = f"{username}/{repo_name}"

    if True:
        # login(token=token)
        # api = HfApi()
        # if create_repo:
        #     api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        assets_readme_path = notebooks_path.parent / "assets" / "README.md"
        readme_text = _load_and_customize_readme(
            assets_readme_path, repo_id, resolutions, split_names, cfg["export"]["default"]
        )
        with open(export_dirpath / "README.md", "w", encoding="utf-8") as fh:
            fh.write(readme_text)
        # _upload_readme(api, repo_id, readme_text)

        # for image_filepath in (notebooks_path.parent / "assets").glob("*.png"):
        #     api.upload_file(
        #         path_or_fileobj=str(image_filepath.resolve()),
        #         path_in_repo=f"assets/{image_filepath.name}",
        #         repo_id=repo_id,
        #         repo_type="dataset",
        #     )
        (export_dirpath / "assets").mkdir(parents=True, exist_ok=True)
        for image_filepath in (notebooks_path.parent / "assets").glob("*.png"):
            shutil.copy2(
                str(image_filepath.resolve()), str((export_dirpath / "assets" / image_filepath.name).resolve())
            )

        # for filepath in calibrate_path.glob("*"):
        #     api.upload_file(
        #         path_or_fileobj=str(filepath.resolve()),
        #         path_in_repo=f"calibration/{filepath.name}",
        #         repo_id=repo_id,
        #         repo_type="dataset",
        #     )
        (export_dirpath / "calibration").mkdir(parents=True, exist_ok=True)
        for filepath in calibrate_path.glob("*"):
            if ".bak" in filepath.name:
                continue
            shutil.copy2(
                str(filepath.resolve()),
                str((export_dirpath / "calibration" / filepath.name).resolve()),
            )

        for res in resolutions:
            for sample in [True, False]:
                config_name = _config_name(res, sample)
                res_dir = dataset_path / str(res)
                if not res_dir.exists():
                    continue

                # split_datasets: dict[str, Dataset] = {}
                for split in split_names:
                    split_dir = res_dir / split
                    if not split_dir.exists():
                        continue
                    export_path = export_dirpath / config_name / split
                    export_path.mkdir(parents=True, exist_ok=True)
                    # TODO: replace here multiprocessing, merging _push_resolution
                    # rows = _load_rows_for_split(split_dir, resolution=res)
                    metadata_root = split_dir / "metadata"
                    # metadata_filepaths = sorted(p for p in metadata_root.rglob("*.yaml") if p.stem.isdigit())
                    metadata_filepaths = list(metadata_root.rglob("*.yaml"))
                    with mp.Pool(processes=max(os.cpu_count() - 1, 1)) as pool:
                        rows: list[dict] = []
                        rows_max = cfg["export"]["sample"]["num_samples"] if sample else None
                        chunk_size = chunk_sizes[res]
                        logger.info(
                            f"Processing {len(metadata_filepaths)} files for {config_name}/{split} with chunk size {chunk_size}"
                        )
                        num_chunk = 0
                        num_chunks = len(metadata_filepaths) // chunk_size + (
                            1 if len(metadata_filepaths) % chunk_size != 0 else 0
                        )
                        for row in tqdm(
                            pool.imap(partial(_process_meta_for_prepare, res, sample), metadata_filepaths),
                            total=len(metadata_filepaths),
                            desc=f"Loading metadata for {config_name}/{split}",
                            unit="file",
                        ):
                            rows.append(row)
                            if len(rows) >= chunk_size:
                                table = Dataset.from_list(rows, features=_features(sample)).data.table
                                parquet_filepath = f"{export_path}/data-{num_chunk:05d}.parquet"
                                logger.info(
                                    f"Writing chunk {num_chunk + 1}/{num_chunks} for {config_name}/{split} with {len(rows)} rows: {parquet_filepath}"
                                )
                                pq.write_table(
                                    table,
                                    parquet_filepath,
                                    compression="zstd",
                                    compression_level=cfg["export"]["compression_level"],
                                )
                                rows = []
                                num_chunk += 1
                            if rows_max is not None and len(rows) >= rows_max:
                                logger.info(
                                    f"Reached maximum sample size of {rows_max} for {config_name}/{split}; stopping load"
                                )
                                break
                        if len(rows) > 0:
                            table = Dataset.from_list(rows, features=_features(sample)).data.table
                            parquet_filepath = f"{export_path}/data-{num_chunk:05d}.parquet"
                            logger.info(
                                f"Writing final chunk for {config_name}/{split} with {len(rows)} rows: {parquet_filepath}"
                            )
                            pq.write_table(
                                table,
                                parquet_filepath,
                                compression="zstd",
                                compression_level=cfg["export"]["compression_level"],
                            )
                            rows = []
                            num_chunk += 1

                #     rows: list[dict] = []
                #     for p in metadata_filepaths:
                #         rows.append(_process_meta_for_prepare(str(p), res))
                #     # return rows
                #     if not rows:
                #         continue
                #     split_datasets[split] = _rows_to_dataset(rows)

                # if not split_datasets:
                #     continue

                # _push_resolution(
                #     api=api,
                #     export_dirpath=export_dirpath,
                #     token=token,
                #     repo_id=repo_id,
                #     resolution=res,
                #     split_datasets=split_datasets,
                #     private=private,
                #     chunk_size=chunk_sizes[res],
                # )

                # _create_parquets(
                #     export_dirpath=export_dirpath,
                #     resolution=res,
                #     split_datasets=split_datasets,
                #     chunk_size=chunk_sizes[res],
                # )

                if execute_notebooks and not sample:
                    output_notebook = export_dirpath / config_name / "dataset.ipynb"
                    output_notebook.parent.mkdir(parents=True, exist_ok=True)
                    parameters = {
                        "RESOLUTION": res,
                        "DATASET_DIRPATH": str(dataset_path),
                        "CONFIG_FILEPATH": str(config_file),
                        "CALIBRATE_FILEPATH": str(calibrate_path / "calibrate.json"),
                        "MAX_WORKERS": os.cpu_count() or 8,
                    }
                    retries = None if max_notebook_retries == 0 else max_notebook_retries
                    _execute_notebook_with_retries(
                        dataset_notebook,
                        output_notebook,
                        parameters=parameters,
                        cwd=notebooks_path,
                        max_retries=retries,
                        retry_delay=notebook_retry_delay,
                    )
                    # _upload_notebook(api, repo_id, output_notebook, res)

        if execute_notebooks:
            for notebook_filepath in notebooks_path.glob("*.ipynb"):
                if notebook_filepath.name.startswith("_"):
                    logger.info(f"Skipping notebook {notebook_filepath.name} (starts with '_')")
                    continue
                output_notebook = export_dirpath / "notebooks" / notebook_filepath.name
                output_notebook.parent.mkdir(parents=True, exist_ok=True)
                retries = None if max_notebook_retries == 0 else max_notebook_retries
                _execute_notebook_with_retries(
                    notebook_filepath,
                    output_notebook,
                    parameters={},
                    cwd=notebooks_path,
                    max_retries=retries,
                    retry_delay=notebook_retry_delay,
                )
                # api.upload_file(
                #     path_or_fileobj=str(output_notebook.resolve()),
                #     path_in_repo=f"notebooks/{notebook_filepath.name}",
                #     repo_id=repo_id,
                #     repo_type="dataset",
                # )

    if skip_upload:
        logger.info("Skipping upload as requested; notebooks (if any) executed")
        return True

    login(token=token)
    api = HfApi()
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(export_dirpath.resolve()),
        num_workers=max(os.cpu_count() - 1, 1),
        ignore_patterns=["*.ipynb_checkpoints"],
    )

    logger.info(f"✓ Export complete: https://huggingface.co/datasets/{repo_id}")
    return True


def _load_and_customize_readme(
    assets_readme_path: Path,
    repo_id: str,
    resolutions: list[int],
    split_names: list[str],
    default_config_name: str,
) -> str:
    """Load README from assets and prepend Hugging Face YAML header with customizations."""

    # Generate the HF YAML header
    yaml_header = _generate_huggingface_yaml_header(repo_id, resolutions, split_names, default_config_name)

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
