"""
Download module for PASSAGE.

This module downloads ALOS AW3D30 tiles by checking URL availability,
tracks the download status in a CSV file, and generates a PNG status map.
"""

import multiprocessing
import os
import zipfile
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm
import shutil

HTTP_STATUS_OK = 200


def download_tile(args: tuple[int, int, str | None]) -> tuple[int, int, str, bool, bool]:
    """
    Check if a URL is valid for a given lat/lon pair, and download/extract if valid.

    Args:
        args: Tuple of (lat_min, lon_min, raw_dir_str) coordinates.
              raw_dir_str can be None to skip download, or a path string for multiprocessing compatibility.

    Returns:
        Tuple of (lat_min, lon_min, url, url_valid, extracted).
    """
    lat_min, lon_min, raw_dir_str = args
    lat_max = lat_min + 5
    lon_max = lon_min + 5
    lat_min_char = "S" if lat_min < 0 else "N"
    lat_max_char = "S" if lat_max < 0 else "N"
    lon_min_char = "W" if lon_min < 0 else "E"
    lon_max_char = "W" if lon_max < 0 else "E"
    part1 = f"{lat_min_char}{abs(lat_min):03d}{lon_min_char}{abs(lon_min):03d}"
    part2 = f"{lat_max_char}{abs(lat_max):03d}{lon_max_char}{abs(lon_max):03d}"
    filename = f"{part1}_{part2}.zip"
    url = f"https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2404/{filename}"

    try:
        response = requests.head(url, timeout=1)
        status_code = response.status_code
        url_valid = status_code == HTTP_STATUS_OK
    except requests.exceptions.RequestException:
        url_valid = False

    extracted = False

    # Download and extract if valid and raw_dir is provided
    if url_valid and raw_dir_str is not None:
        raw_dir = Path(raw_dir_str)
        try:
            zip_path = raw_dir / "archives" / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            # extract_folder = raw_dir / "extracted"
            # extract_folder.mkdir(parents=True, exist_ok=True)
            files_folder = raw_dir / "tiles"
            files_folder.mkdir(parents=True, exist_ok=True)

            # Download the file once
            response = requests.get(url, timeout=180, stream=True)
            if response.status_code == HTTP_STATUS_OK:
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract only *_DSM.tif entries into files_folder without extracting whole archive
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        members = [m for m in zip_ref.namelist() if m.upper().endswith("_DSM.TIF")]
                        if members:
                            count = 0
                            for member in members:
                                filename_member = os.path.basename(member)
                                if not filename_member:
                                    continue
                                dest_path = files_folder / filename_member
                                try:
                                    with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                                        shutil.copyfileobj(src, dst)
                                    count += 1
                                except Exception as e:
                                    logger.warning(f"Failed copying DSM file {member} to {dest_path}: {e}")
                            if count > 0:
                                extracted = True
                                logger.info(f"Copied {count} _DSM.tif files from {zip_path.name} to {files_folder}")
                        else:
                            logger.info(f"No _DSM.tif files found in archive {zip_path.name}")
                except Exception as e:
                    logger.error(f"Failed to extract DSM files from archive {zip_path}: {e}")

                # Cleanup: remove zip file
                try:
                    zip_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove zip file {zip_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to download or extract tile for {lat_min}, {lon_min}: {e}")
            pass  # Continue even if download fails, we still return validity

    return (lat_min, lon_min, url, url_valid, extracted)


def generate_status_map(df_results: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a PNG visualization of URL status.

    Args:
        df_results: DataFrame with lat_min, lon_min, url, url_valid columns.
        output_path: Path to save the PNG file.
    """
    logger.info("Generating status map image...")

    # Define geographic parameters
    MIN_LAT_DATA = -90
    MAX_LAT_DATA = 85
    MIN_LON_DATA = -180
    MAX_LON_DATA = 175
    LAT_STEP = 5
    LON_STEP = 5

    NUM_LAT_STEPS = int((MAX_LAT_DATA - MIN_LAT_DATA) / LAT_STEP) + 1
    NUM_LON_STEPS = int((MAX_LON_DATA - MIN_LON_DATA) / LON_STEP) + 1

    CELL_SIZE = 10  # Size of each cell in pixels
    IMG_HEIGHT = NUM_LAT_STEPS * CELL_SIZE
    IMG_WIDTH = NUM_LON_STEPS * CELL_SIZE

    # Create a blank image (black background). OpenCV uses BGR color format.
    image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    COLOR_WHITE_BGR = (255, 255, 255)  # Valid URL (White)
    COLOR_BLACK_BGR = (0, 0, 0)  # Invalid URL (Black)

    if not df_results.empty:
        for _index, row in df_results.iterrows():
            lat_min = row["lat_min"]
            lon_min = row["lon_min"]
            url_valid = row["url_valid"]

            # Normalize latitude to y-index: higher latitudes (north) at smaller y (top)
            y_idx = int((MAX_LAT_DATA - lat_min) / LAT_STEP)

            # Normalize longitude to x-index: lower longitudes (west) at smaller x (left)
            x_idx = int((lon_min - MIN_LON_DATA) / LON_STEP)

            color = COLOR_WHITE_BGR if url_valid else COLOR_BLACK_BGR

            # Calculate pixel coordinates for the top-left corner of the cell
            y1 = y_idx * CELL_SIZE
            x1 = x_idx * CELL_SIZE

            # Assign color to the pixels in the cell
            if 0 <= y1 < IMG_HEIGHT and 0 <= x1 < IMG_WIDTH:
                y2 = min(y1 + CELL_SIZE, IMG_HEIGHT)
                x2 = min(x1 + CELL_SIZE, IMG_WIDTH)
                image[y1:y2, x1:x2] = color
            else:
                logger.warning(
                    f"Calculated cell indices ({y_idx}, {x_idx}) "
                    f"for lat/lon ({lat_min}, {lon_min}) are out of "
                    "expected image bounds. Skipping."
                )
    else:
        logger.warning("DataFrame is empty. Cannot generate status map image.")

    try:
        cv2.imwrite(str(output_path), image)
        logger.info(f"Status map image saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save image {output_path}: {e}")


def run_download(outputs_dirpath: str, *, resume: bool = True, num_cpus: int | None = None) -> None:
    """
    Main orchestration function for downloading and tracking tile availability.

    Args:
        outputs_dirpath: Directory path for output files.
        resume: If True, resume from existing CSV file, skipping already-checked pairs.
        num_cpus: Number of CPUs to use for multiprocessing.
            None: use all CPUs (default)
            0: disable multiprocessing (useful for debugging)
            N: use N CPUs
    """
    outputs_path = Path(outputs_dirpath)
    outputs_path.mkdir(parents=True, exist_ok=True)

    # Harmonized startup log
    cpus = num_cpus if num_cpus is not None else os.cpu_count()
    logger.info(f"Starting download: outputs={outputs_path}, cpus={cpus}, resume={resume}")

    csv_path = outputs_path / "download.csv"
    png_path = outputs_path / "world_coverage.png"

    # Generate all lat/lon pairs
    all_pairs = []
    for lat_min in range(-90, 90, 5):
        for lon_min in range(-180, 180, 5):
            all_pairs.append((lat_min, lon_min))

    # Load existing results if resume is enabled
    existing_results: list[tuple[int, int, str, bool]] = []
    existing_pairs: set[tuple[int, int]] = set()
    existing_extracted: dict[tuple[int, int], bool] = {}

    if resume and csv_path.exists():
        logger.info(f"Resuming from existing CSV file: {csv_path}")
        try:
            df_existing = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df_existing)} existing results")
            for _, row in df_existing.iterrows():
                lat, lon = int(row["lat_min"]), int(row["lon_min"])
                extracted_flag = bool(row.get("extracted", False))
                existing_pairs.add((lat, lon))
                existing_extracted[(lat, lon)] = extracted_flag
                existing_results.append((lat, lon, row["url"], row["url_valid"], extracted_flag))
        except (pd.errors.EmptyDataError, Exception) as e:
            logger.warning(f"Failed to load CSV: {e}. Starting fresh.")
    else:
        if csv_path.exists():
            os.unlink(csv_path)
        logger.info("Starting fresh download (no resume)")

    # Create raw directory for downloads
    raw_dir = outputs_path
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dir_str = str(raw_dir)

    # Filter pairs to process. If resuming, re-queue pairs whose previous 'extracted' value was False
    pairs_to_process = []
    for lat, lon in all_pairs:
        if (lat, lon) not in existing_pairs:
            pairs_to_process.append((lat, lon, raw_dir_str))
        else:
            # if previously checked but not extracted successfully, redo
            if not existing_extracted.get((lat, lon), False):
                pairs_to_process.append((lat, lon, raw_dir_str))

    if pairs_to_process:
        num_already_checked = len(existing_pairs)
        num_failed_extractions = sum(1 for v in existing_extracted.values() if not v)
        logger.info(
            f"Processing {len(pairs_to_process)} remaining pairs (skipping {num_already_checked} already checked; "
            f"reprocessing {num_failed_extractions} failed extractions)"
        )

        file_exists = csv_path.exists()
        new_results = []

        # Open CSV once and append each result as it is produced
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # write header if file did not exist before
            if not file_exists:
                writer.writerow(["lat_min", "lon_min", "url", "url_valid", "extracted"])
                csvfile.flush()
                try:
                    os.fsync(csvfile.fileno())
                except Exception:
                    pass

            # Process URL checks and downloads, writing each result immediately
            if num_cpus == 0:
                # Sequential processing (useful for debugging)
                logger.info("Running in sequential mode (no multiprocessing)")
                for res in tqdm((download_tile(pair) for pair in pairs_to_process), total=len(pairs_to_process)):
                    try:
                        writer.writerow(res)
                        csvfile.flush()
                        try:
                            os.fsync(csvfile.fileno())
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"Failed writing result {res} to CSV: {e}")
                    new_results.append(res)
            else:
                # Multiprocessing
                pool_size = num_cpus if num_cpus is not None else os.cpu_count()
                logger.info(f"Running with {pool_size} CPUs")
                with multiprocessing.Pool(pool_size) as pool:
                    for res in tqdm(pool.imap(download_tile, pairs_to_process), total=len(pairs_to_process)):
                        try:
                            writer.writerow(res)
                            csvfile.flush()
                            try:
                                os.fsync(csvfile.fileno())
                            except Exception:
                                pass
                        except Exception as e:
                            logger.warning(f"Failed writing result {res} to CSV: {e}")
                        new_results.append(res)

        # Merge existing and new results
        all_results = existing_results + new_results
        df_results = pd.DataFrame(all_results, columns=["lat_min", "lon_min", "url", "url_valid", "extracted"])
        logger.info(f"URL status codes appended to {csv_path}")
    else:
        logger.info("All pairs already checked, nothing to process")
        df_results = pd.DataFrame(existing_results, columns=["lat_min", "lon_min", "url", "url_valid", "extracted"])

    # Generate status map
    generate_status_map(df_results, png_path)
    logger.info("Download completed.")
