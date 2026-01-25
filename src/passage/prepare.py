"""Prepare module for PASSAGE.

Creates a Hugging Face-friendly export structure under outputs/export with
chunked parquet shards per resolution and split, plus documentation notebooks.
Supports direct upload to Hugging Face Hub with optimal compression.
"""

import io
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from pathlib import Path

try:
    import papermill as pm  # type: ignore

    PAPERMILL_AVAILABLE = True
except Exception:
    pm = None
    PAPERMILL_AVAILABLE = False
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from loguru import logger
from PIL import Image
from tqdm import tqdm


def _parse_path_txt(txt_path: Path) -> list[list[int]]:
    """Parse a path .txt file written as: "0 x0 y0 x1 y1 ..." and return [[i,j], ...]."""
    txt = txt_path.read_text().strip()
    if not txt:
        return []
    parts = txt.split()
    if len(parts) < 3:
        return []
    coords = list(map(int, parts[1:]))
    # stored as (x0 y0 x1 y1 ...) where x=col, y=row -> return [[row,col], ...]
    return [[coords[i + 1], coords[i]] for i in range(0, len(coords) - 1, 2)]


def _process_meta_for_prepare(metadata_filepath: str) -> dict | None:
    """Process a single metadata file into a serializable dict for parquet writing.

    Returns a dict with keys: id, tensor (bytes|None), path_free (list), path_obstacles (list), image (bytes|None), metadata (dict)
    """
    # Build related file paths
    metadata_filepath = Path(metadata_filepath)
    split_dirpath = Path(metadata_filepath.parent.parent.parent.parent)
    lat_dir = metadata_filepath.parent.parent.name
    lon_dir = metadata_filepath.parent.name
    stem = metadata_filepath.stem
    tensor_filepath = split_dirpath / "tensors" / lat_dir / lon_dir / f"{stem}.npy.zst"
    image_filepath = split_dirpath / "images" / lat_dir / lon_dir / f"{stem}.png"
    path_free_filepath = split_dirpath / "paths" / "free" / lat_dir / lon_dir / f"{stem}.txt"
    path_obstacles_filepath = split_dirpath / "paths" / "obstacles" / lat_dir / lon_dir / f"{stem}.txt"

    metadata = yaml.safe_load(metadata_filepath.read_text())

    image = None
    try:
        if image_filepath.exists():
            with Image.open(image_filepath) as im:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                image = buf.getvalue()
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to load image {image_filepath}: {exc}")
        # Fallback: try raw bytes if available
        try:
            image = image_filepath.read_bytes()
        except Exception:
            image = None

    # Return a ready-to-serialize Python dict matching the parquet columns used later
    return {
        "image": image,
        "id": str(stem),
        "tensor": tensor_filepath.read_bytes(),
        "path_free": _parse_path_txt(path_free_filepath),
        "path_obstacles": _parse_path_txt(path_obstacles_filepath),
        "metadata": metadata,
    }


def _write_parquet_shard(
    rows: list[dict],
    shard_path: Path,
    resolution: int,
    images_dir: Path | None = None,
) -> None:
    """Write one Parquet shard using a list of Python dicts and pa.Table.from_pylist.

    This keeps the code simpler and lets pyarrow infer appropriate nested types for
    lists and structs. Ensure that paths are plain Python ints so Arrow will infer
    nested list<int32> for the path columns.

    Images are stored as {"bytes": <raw_png_bytes>, "path": <str|None>} struct to be
    compatible with Hugging Face datasets Image feature type. When `images_dir` is
    provided, the image bytes will additionally be written to disk and the "path"
    field will point to a repo-relative path (e.g. "images/256/train/ID.png") so
    that Hugging Face Data Studio can render the images.

    Compression uses zstd level 19 for best compression ratio while maintaining
    reasonable decompression speed.
    """

    if not rows:
        return

    def _normalize_paths(paths: list[list[int]]) -> list[list[int]]:
        return [[int(x) for x in sub] for sub in paths]

    def _format_image_for_hf(image_bytes: bytes | None, id_str: str) -> dict | None:
        """Format image bytes as HF datasets Image feature struct and optionally write file.

        If `images_dir` is provided, the image will be written to disk at
        `images_dir / <id>.png` and the returned struct will set `path` to the
        repo-relative path starting from the directory named "images" (so it works
        after the export is uploaded to the Hub).
        """
        if image_bytes is None:
            return None

        path_value = None
        # Use a local variable to avoid shadowing the outer `images_dir` name
        images_dir_local = images_dir
        if images_dir_local is not None:
            try:
                images_dir_local.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If we cannot create the dir, continue without writing
                images_dir_local = None

        fname = f"{id_str}.png"
        if images_dir_local is not None:
            out_path = images_dir_local / fname
            try:
                # Avoid rewriting already written files
                if not out_path.exists():
                    out_path.write_bytes(image_bytes)
            except Exception:
                # Failed to write, but still return bytes so HF datasets can decode
                pass

            # Build a repo-relative path starting from the "images" segment so
            # that Data Studio can locate the file after upload (e.g.
            # "images/256/train/123.png"). If "images" is not present in the
            # path parts, fallback to using the filename only.
            parts = list(images_dir_local.parts)
            if "images" in parts:
                idx = parts.index("images")
                rel_parts = parts[idx:]
                path_value = "/".join(rel_parts + [fname])
            else:
                path_value = fname

        return {"bytes": image_bytes, "path": path_value}

    py_rows: list[dict] = []
    for r in rows:
        id_str = str(r.get("id"))
        py_rows.append(
            {
                "image": _format_image_for_hf(r.get("image"), id_str),
                "id": id_str,
                "tensor": r.get("tensor"),
                "path_free": _normalize_paths(r.get("path_free", [])),
                "path_obstacles": _normalize_paths(r.get("path_obstacles", [])),
                "metadata": r.get("metadata"),
            }
        )

    table = pa.Table.from_pylist(py_rows)
    # Use zstd compression level 19 for best compression ratio
    # Level 19 provides excellent compression with acceptable speed
    pq.write_table(
        table,
        shard_path,
        compression="zstd",
        compression_level=19,
        write_statistics=True,
    )


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
    if PAPERMILL_AVAILABLE and pm is not None:
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
    notebook_name = input_notebook.name
    while True:
        attempt += 1
        resolution_info = f" (resolution={parameters.get('RESOLUTION')})" if "RESOLUTION" in parameters else ""
        logger.info(f"Executing notebook {notebook_name} (attempt {attempt}){resolution_info}")
        success = _execute_notebook_with_papermill(input_notebook, output_notebook, parameters, cwd)
        if success:
            return True

        # Execution failed; decide whether to retry
        if max_retries is None:
            logger.warning(f"Notebook {notebook_name} execution failed; retrying indefinitely.")
            time.sleep(max(0, int(retry_delay)))
            continue

        if attempt >= max_retries:
            logger.warning(f"Notebook {notebook_name} execution failed after {attempt} attempt(s).")
            return False

        time.sleep(max(0, int(retry_delay)))


def _generate_huggingface_yaml_header(
    repo_id: str,
    resolutions: list[int],
    split_names: list[str],
) -> str:
    """Generate the Hugging Face YAML header for the dataset README.

    Includes dataset_info with Image feature type to enable visualization
    in Hugging Face Data Studio.
    """

    configs_yaml = []
    for res in sorted(resolutions):
        split_entries = []
        for split in sorted(split_names):
            split_entries.append(f'      - split: {split}\n        path: "{res}/{split}/data-*.parquet"')
        # Use config names like "64x64", "128x128" so subsets are named by resolution
        configs_yaml.append(f'  - config_name: "{res}x{res}"\n    data_files:\n' + "\n".join(split_entries))

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
dataset_info:
  features:
    - name: image
      dtype: image
    - name: id
      dtype: string
    - name: tensor
      dtype: binary
    - name: path_free
      sequence:
        sequence:
          dtype: int32
    - name: path_obstacles
      sequence:
        sequence:
          dtype: int32
    - name: metadata
      dtype: string
configs:
{os.linesep.join(configs_yaml)}
---

"""


def _load_hf_credentials(
    env_path: Path | None = None,
    secret_path: Path | None = None,
    require_token: bool = True,
) -> tuple[str, str, str]:
    """Load Hugging Face credentials from .env and .env.secret files.

    Args:
        env_path: Path to .env file (default: project root .env)
        secret_path: Path to .env.secret file (default: project root .env.secret)
        require_token: Whether to raise error if token is missing (default: True)

    Returns:
        Tuple of (username, repo_name, token)

    Raises:
        ValueError: If required credentials are missing
    """
    # Find project root (where .env files are located)
    project_root = Path(__file__).parent.parent.parent

    if env_path is None:
        env_path = project_root / ".env"
    if secret_path is None:
        secret_path = project_root / ".env.secret"

    # Load .env file for username and repo name
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")

    # Load .env.secret file for API token
    if secret_path.exists():
        load_dotenv(secret_path)
        logger.debug(f"Loaded secrets from {secret_path}")
    else:
        if require_token:
            logger.warning(f".env.secret file not found at {secret_path}")

    # Get credentials from environment
    username = os.getenv("HF_USER_NAME", "")
    repo_name = os.getenv("HF_REPO_NAME", "")
    token = os.getenv("HF_TOKEN", "")

    if not username:
        raise ValueError("HF_USER_NAME not found in .env file")
    if not repo_name:
        raise ValueError("HF_REPO_NAME not found in .env file")
    if require_token and not token:
        raise ValueError("HF_TOKEN not found in .env.secret file")

    return username, repo_name, token


def upload_to_huggingface(
    export_dirpath: str | Path = "outputs/export",
    env_path: str | Path | None = None,
    secret_path: str | Path | None = None,
    repo_id: str | None = None,
    private: bool = False,
    create_repo: bool = True,
) -> bool:
    """Upload prepared export to Hugging Face Hub.

    Credentials are loaded from .env (HF_USER_NAME, HF_REPO_NAME) and
    .env.secret (HF_TOKEN) files unless repo_id is explicitly provided.

    Args:
        export_dirpath: Path to prepared export directory
        env_path: Optional path to .env file
        secret_path: Optional path to .env.secret file
        repo_id: Optional explicit repo_id (overrides .env settings)
        private: Whether the repository should be private
        create_repo: Whether to create the repository if it doesn't exist

    Returns:
        True if upload succeeded, False otherwise
    """
    export_path = Path(export_dirpath).resolve()

    if not export_path.exists():
        logger.error(f"Export directory not found: {export_path}")
        logger.info("Run 'passage prepare' first to create the export structure")
        return False

    # Load credentials
    try:
        username, repo_name, token = _load_hf_credentials(
            env_path=Path(env_path) if env_path else None,
            secret_path=Path(secret_path) if secret_path else None,
        )
    except ValueError as e:
        logger.error(f"Failed to load HF credentials: {e}")
        return False

    # Use explicit repo_id or construct from .env
    if repo_id is None:
        repo_id = f"{username}/{repo_name}"

    logger.info(f"Uploading to Hugging Face: {repo_id}")

    # Login to Hugging Face
    try:
        login(token=token)
        logger.info("✓ Authenticated with Hugging Face Hub")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False

    # Initialize HfApi
    api = HfApi()

    # Create repository if requested
    if create_repo:
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            logger.info(f"✓ Repository ready: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False

    # Remove any lingering 'default' config in dataset metadata if present to
    # avoid having a legacy "default" subset on the Hub. This is best-effort.
    # FIXME: to execute only if there is a default subset
    # try:
    #     repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    #     if "dataset_infos.json" in repo_files:
    #         # Download, edit out 'default' key if present, and re-upload
    #         import json

    #         from huggingface_hub import hf_hub_download

    #         try:
    #             ds_infos_path = hf_hub_download(repo_id=repo_id, filename="dataset_infos.json", repo_type="dataset")
    #             with open(ds_infos_path, "r", encoding="utf-8") as fh:
    #                 ds_infos = json.load(fh)
    #             if "default" in ds_infos:
    #                 logger.info("Removing legacy 'default' configuration from dataset_infos.json")
    #                 ds_infos.pop("default", None)
    #                 api.upload_file(
    #                     path_or_fileobj=json.dumps(ds_infos).encode("utf-8"),
    #                     path_in_repo="dataset_infos.json",
    #                     repo_id=repo_id,
    #                     repo_type="dataset",
    #                 )
    #         except Exception:
    #             logger.debug("Could not remove legacy 'default' config (non-fatal)")
    # except Exception:
    #     logger.debug("Could not inspect remote repo files for default config (non-fatal)")

    # Upload non-parquet files (assets, notebooks, README). Parquet shards will be
    # uploaded via `datasets.push_to_hub` below to ensure proper config naming
    # and image embedding. This prevents a legacy 'default' config from being
    # automatically created by a raw folder upload.
    try:
        logger.info(f"Uploading non-parquet files from {export_path}...")
        api.upload_folder(
            folder_path=str(export_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload PASSAGE dataset (assets & docs)",
            ignore_patterns=[
                "*.pyc",
                "__pycache__",
                ".git",
                ".gitignore",
                "*.log",
                "*.parquet",
            ],
        )
        logger.info("✓ Non-parquet upload complete!")
    except Exception as e:
        logger.error(f"Failed to upload non-parquet files to Hugging Face: {e}")
        return False

    # Upload parquet shards and push dataset metadata per resolution
    try:
        logger.info("Pushing parquet shards and dataset metadata (per-resolution configs) to the Hub...")
        pushed = _push_dataset_metadata_to_hub(repo_id, export_dirpath, resolutions=None, private=private)
        if not pushed:
            logger.warning(
                "Could not push dataset metadata to Hub. The non-parquet assets were uploaded, but Data Studio may still show 'dict' until metadata is updated manually."
            )

        # Upload README.md to overwrite the dataset card with our customized README
        try:
            readme_path = export_path / "README.md"
            if readme_path.exists():
                api.upload_file(
                    path_or_fileobj=readme_path.read_bytes(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info("✓ Uploaded custom README to dataset repo")
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"Failed to upload README to dataset repo: {exc}")

        logger.info("✓ Upload complete!")
        logger.info(f"  Dataset URL: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        return False


def _push_dataset_metadata_to_hub(
    repo_id: str,
    export_dirpath: str | Path = "outputs/export",
    resolutions: list[int] | None = None,
    private: bool = False,
) -> bool:
    """Load parquet shards, cast features (Image) and push dataset metadata to HF Hub.

    This helps ensure Data Studio recognizes the `image` column as an Image feature
    so it renders correctly. If `datasets` package is not available, logs a
    warning and returns False.
    """
    try:
        from datasets import Features, Image, Sequence, Value, load_dataset
    except Exception:
        logger.warning(
            "`datasets` package not available; cannot push dataset metadata to Hub. Install with `pip install datasets`"
        )
        return False

    export_path = Path(export_dirpath).resolve()
    if not export_path.exists():
        logger.error(f"Export path not found: {export_path}")
        return False

    # If resolutions not provided, attempt to infer
    if resolutions is None:
        resolutions = [int(d.name) for d in export_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if not resolutions:
        logger.warning("No resolutions found under export to build dataset metadata")
        return False

    # Use first resolution with available parquet files
    used = False
    for res in resolutions:
        data_dir = export_path / str(res)
        if not data_dir.exists():
            continue

        # Build data_files mapping for dataset.load_dataset
        data_files = {}
        for split in ["train", "validation", "test", "calibration"]:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            files = sorted([str(p) for p in split_dir.glob("data-*.parquet")])
            if files:
                data_files[split] = files

        if not data_files:
            continue

        logger.info(
            f"Building HF dataset metadata from resolution {res} (files: {sum(len(v) for v in data_files.values())})"
        )
        try:
            # Use a named config (e.g. "64x64") so that each resolution appears as its own
            # dataset config/subset on Hugging Face rather than a single "default" config.
            config_name = f"{res}x{res}"
            ds = load_dataset("parquet", name=config_name, data_files=data_files)
        except Exception as exc:
            logger.warning(f"Failed to load parquet for resolution {res}: {exc}")
            continue

        # Some columns are complex nested structs (e.g. 'paths', 'tile', 'timing') which
        # cannot be cast to a plain string type directly. Convert all non-essential
        # nested columns to JSON strings so we can safely cast and publish metadata.
        try:
            import json

            # Determine columns to keep as-is (non-stringified)
            keep_cols = {"image", "id", "tensor", "path_free", "path_obstacles"}

            # Build a set of columns present in the dataset (use train split as representative)
            any_split = next(iter(ds.keys()))
            all_cols = set(ds[any_split].column_names)

            cols_to_stringify = [c for c in all_cols if c not in keep_cols]

            if cols_to_stringify:
                logger.debug(f"Stringifying nested columns before casting: {cols_to_stringify}")

                def _stringify(batch):
                    out = {}
                    for k, v in batch.items():
                        if k in cols_to_stringify:
                            # JSON-encode nested items; preserve None
                            out[k] = [json.dumps(x, default=str) if x is not None else None for x in v]
                        else:
                            out[k] = v
                    return out

                ds = ds.map(_stringify, batched=True)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"Failed to stringify nested columns: {exc}")

        features = Features(
            {
                "image": Image(),
                "id": Value("string"),
                "tensor": Value("binary"),
                "path_free": Sequence(Sequence(Value("int32"))),
                "path_obstacles": Sequence(Sequence(Value("int32"))),
                "metadata": Value("string"),
            }
        )

        try:
            ds = ds.cast(features)
        except Exception as exc:
            logger.warning(f"Failed to cast dataset features: {exc}")

        try:
            # Use the datasets library to push the dataset as a named config so
            # Hugging Face will present each resolution as its own subset instead
            # of creating/using a "default" config.
            ds.push_to_hub(repo_id, config_name=config_name, private=private, set_default=False)
            logger.info(f"✓ Pushed dataset metadata to hub for config: {config_name} -> {repo_id}")
            used = True
            # Continue attempting other resolutions so all configs are published
            continue
        except Exception as exc:
            logger.warning(f"Failed to push dataset metadata to hub for config {config_name}: {exc}")
            continue

    if not used:
        logger.warning("Could not push dataset metadata to Hub for any resolution")
        return False

    return True


def export_and_upload(
    export_dirpath: str | Path = "outputs/export",
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    notebooks_dirpath: str | Path = "notebooks",
    resolutions: list[int] | None = None,
    repo_id: str | None = None,
    execute_notebooks: bool = True,
    max_notebook_retries: int | None = 3,
    notebook_retry_delay: int = 5,
    num_cpus: int | None = None,
    env_path: str | Path | None = None,
    secret_path: str | Path | None = None,
    private: bool = False,
    create_repo: bool = True,
    skip_upload: bool = False,
) -> bool:
    """Prepare export and optionally upload to Hugging Face, then ensure dataset metadata.

    If `skip_upload` is True, only runs `prepare_export` without uploading. Otherwise,
    it uploads and attempts to publish dataset metadata with Image features so Data
    Studio renders images.
    """
    ok = prepare_export(
        export_dirpath=export_dirpath,
        dataset_dirpath=dataset_dirpath,
        calibrate_dirpath=calibrate_dirpath,
        notebooks_dirpath=notebooks_dirpath,
        resolutions=resolutions,
        repo_id=repo_id,
        execute_notebooks=execute_notebooks,
        max_notebook_retries=max_notebook_retries,
        notebook_retry_delay=notebook_retry_delay,
        num_cpus=num_cpus,
    )

    if not ok:
        logger.error("Export preparation failed; aborting upload")
        return False

    if skip_upload:
        logger.info("Skipping upload as requested (--no-upload)")
        return True

    # Upload to HF
    uploaded = upload_to_huggingface(
        export_dirpath=export_dirpath,
        env_path=env_path,
        secret_path=secret_path,
        repo_id=repo_id,
        private=private,
        create_repo=create_repo,
    )

    if not uploaded:
        logger.error("Upload failed")
        return False

    # Attempt to push dataset metadata (forces image dtype)
    try:
        # Determine final repo_id to use for pushing dataset metadata
        final_repo_id = repo_id
        if not final_repo_id:
            try:
                username, repo_name, _ = _load_hf_credentials(require_token=False)
                final_repo_id = f"{username}/{repo_name}"
            except ValueError:
                final_repo_id = ""

        # Try to get resolutions list if not provided
        res_list = resolutions
        if res_list is None:
            export_path = Path(export_dirpath).resolve()
            res_list = [int(d.name) for d in export_path.iterdir() if d.is_dir() and d.name.isdigit()]

        if not final_repo_id:
            logger.warning("No repo_id available for pushing dataset metadata to Hub; skipping metadata push.")
            pushed = False
        else:
            pushed = _push_dataset_metadata_to_hub(final_repo_id, export_dirpath, resolutions=res_list, private=private)
        if not pushed:
            logger.warning(
                "Could not push dataset metadata to Hub. The upload itself succeeded, but Data Studio may still show 'dict' until metadata is updated manually."
            )
    except Exception as exc:
        logger.warning(f"Exception while pushing dataset metadata: {exc}")

    return True


def _load_and_customize_readme(
    assets_readme_path: Path,
    repo_id: str,
    resolutions: list[int],
    split_names: list[str],
) -> str:
    """Load README from assets and prepend Hugging Face YAML header with customizations."""

    # Generate the HF YAML header
    yaml_header = _generate_huggingface_yaml_header(repo_id, resolutions, split_names)

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


def prepare_export(
    export_dirpath: str | Path = "outputs/export",
    dataset_dirpath: str | Path = "outputs/dataset",
    calibrate_dirpath: str | Path = "outputs/calibrate",
    notebooks_dirpath: str | Path = "notebooks",
    resolutions: list[int] | None = None,
    repo_id: str | None = None,
    execute_notebooks: bool = True,
    max_notebook_retries: int | None = 3,
    notebook_retry_delay: int = 5,
    num_cpus: int | None = None,
) -> bool:
    """Prepare export artifacts in outputs/export (no upload).

    If repo_id is not provided, it will be loaded from .env file
    (HF_USER_NAME/HF_REPO_NAME).

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

    # Load repo_id from .env if not explicitly provided
    if repo_id is None or repo_id == "username/passage":
        try:
            username, repo_name, _ = _load_hf_credentials(require_token=False)
            repo_id = f"{username}/{repo_name}"
            logger.info(f"Using repo_id from .env: {repo_id}")
        except ValueError:
            # Fall back to default if .env is not configured
            repo_id = "username/passage"
            logger.warning("Could not load HF credentials from .env, using default repo_id")

    calibrate_path = Path(calibrate_dirpath).resolve()
    notebooks_path = Path(notebooks_dirpath).resolve()

    # Load export chunk sizes from config.yaml (rows per shard). Keep it simple.
    config_file = notebooks_path.parent / "config" / "config.yaml"
    config = yaml.safe_load(config_file.read_text())
    chunk_map = config["export"]["chunk_size"]

    if resolutions is None:
        resolutions = [int(d.name) for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if not resolutions:
        logger.warning("No resolution directories found to prepare")
        return False

    export_path.mkdir(parents=True, exist_ok=True)

    # Copy assets folder (images and README template)
    assets_path = notebooks_path.parent / "assets"
    assets_export = export_path / "assets"
    if assets_path.exists():
        if assets_export.exists():
            shutil.rmtree(assets_export)
        shutil.copytree(assets_path, assets_export)
        logger.info(f"✓ Copied assets folder to {assets_export}")
        # Ensure sample.png is present in the exported assets directory (fallback behavior)
        sample_src = assets_path / "sample.png"
        sample_dst = assets_export / "sample.png"
        if not sample_dst.exists():
            if sample_src.exists():
                shutil.copy2(sample_src, sample_dst)
                logger.info(f"✓ Copied sample.png to {sample_dst}")
            else:
                alt_sample = notebooks_path.parent / "sample.png"
                if alt_sample.exists():
                    shutil.copy2(alt_sample, sample_dst)
                    logger.info(f"✓ Copied sample.png from project root to {sample_dst}")

    # Execute and export notebooks to outputs/export/notebooks
    notebooks_export = export_path / "notebooks"
    notebooks_export.mkdir(exist_ok=True)

    # Common paths for notebook execution - notebooks use ROOT = Path("..").resolve()
    # so when executed from notebooks_path as cwd, they automatically find the right paths.
    # We only need to override scalar parameters that can be injected via papermill.
    config_path = notebooks_path.parent / "config" / "config.yaml"
    calibrate_json_path = calibrate_path / "calibrate.json"
    max_workers_param = os.cpu_count() or 8

    # Define notebook parameters for each notebook
    # Note: Path variables are computed in notebooks relative to ROOT, which works
    # when executed with cwd=notebooks_path. We only override scalar values.
    notebook_params = {
        "costmodel.ipynb": {
            "RESOLUTION": 512,
            "NUM_SAMPLES": 20,
        },
        "display.ipynb": {
            "RESOLUTION": 512,
            "SPLIT": "train",
            "SAMPLE_INDEX": 0,
        },
        "metrics.ipynb": {
            "RESOLUTION": 512,
        },
        "obstacles.ipynb": {
            "RESOLUTION": 512,
            "SAMPLE_INDEX": 0,
            "DISABLE_IMAGES": False,
            "NUM_MASKS_TO_GENERATE": 30,
        },
        "parquet.ipynb": {
            "RESOLUTION": 512,
            "SPLIT": "train",
            "ROW_INDEX": 0,
        },
        "solve.ipynb": {
            "RESOLUTION": 512,
        },
    }

    # Execute each notebook (except dataset.ipynb which is handled per-resolution)
    if notebooks_path.exists() and execute_notebooks:
        for nb in notebooks_path.glob("*.ipynb"):
            if nb.name == "dataset.ipynb":
                continue  # Handled separately per resolution

            output_notebook = notebooks_export / nb.name
            params = notebook_params.get(nb.name, {})

            logger.info(f"Executing notebook: {nb.name}")
            retries = None if max_notebook_retries == 0 else max_notebook_retries

            success = _execute_notebook_with_retries(
                nb,
                output_notebook,
                parameters=params,
                cwd=notebooks_path,
                max_retries=retries,
                retry_delay=notebook_retry_delay,
            )

            if not success:
                logger.warning(f"Notebook {nb.name} execution failed; copying without execution")
                shutil.copy2(nb, output_notebook)
            else:
                logger.info(f"✓ Notebook executed: {nb.name}")
    elif notebooks_path.exists():
        # Just copy notebooks without execution
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

        # Execute dataset.ipynb for this resolution
        if dataset_notebook.exists():
            output_notebook = res_export / "dataset.ipynb"
            if execute_notebooks:
                parameters = {
                    "RESOLUTION": res,
                    "DATASET_DIR": str(dataset_path),
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

                # Always ensure an output notebook exists in the export so docs are present
                try:
                    shutil.copy2(dataset_notebook, output_notebook)
                except Exception as e:  # pragma: no cover - copying should rarely fail
                    logger.warning(f"Failed copying dataset notebook to export: {e}")

                if not success:
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

            metadata_root = split_src / "metadata"

            split_export = res_export / split
            split_export.mkdir(parents=True, exist_ok=True)

            # Prepare an images output directory (repo path: images/<resolution>/<split>/)
            images_outdir = export_path / "images" / str(res) / split
            images_outdir.mkdir(parents=True, exist_ok=True)

            if not metadata_root.exists():
                logger.warning(f"No metadata for split {split} at {metadata_root}, skipping")
                continue

            shard_size = chunk_map.get(res)
            shard_rows: list[dict] = []
            shard_index = 0
            total_rows = 0

            metadata_filepaths = sorted([p for p in metadata_root.rglob("*.yaml") if p.stem.isdigit()])
            if not metadata_filepaths:
                logger.info(f"No metadata files for split {split} at {metadata_root}, skipping")
                continue

            num_workers = max(1, (num_cpus if num_cpus is not None else os.cpu_count()) - 1)
            chunksize = min(max(1, len(metadata_filepaths) // (num_workers)), 1000)

            with mp.Pool(processes=num_workers) as exe:
                for result in tqdm(
                    exe.imap_unordered(
                        _process_meta_for_prepare,
                        (str(p) for p in metadata_filepaths),
                        chunksize=chunksize,
                    ),
                    total=len(metadata_filepaths),
                ):
                    # Append the prepared row directly (it's already normalized for parquet)
                    shard_rows.append(result)
                    total_rows += 1

                    if len(shard_rows) >= shard_size:
                        shard_path = split_export / f"data-{shard_index:05d}.parquet"
                        _write_parquet_shard(shard_rows, shard_path, res, images_dir=images_outdir)
                        shard_rows.clear()
                        shard_index += 1

            # Flush remaining rows
            if shard_rows:
                shard_path = split_export / f"data-{shard_index:05d}.parquet"
                _write_parquet_shard(shard_rows, shard_path, res, images_dir=images_outdir)
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

    # Generate README from assets template with HF YAML header and customizations
    assets_readme_path = notebooks_path.parent / "assets" / "README.md"
    readme = _load_and_customize_readme(assets_readme_path, repo_id, resolutions, split_names)
    (export_path / "README.md").write_text(readme)

    return True
