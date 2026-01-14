"""
Calibrate module for PASSAGE.

This module scans all ALOS AW3D30 DSM tiles to find global min/max elevation values.
"""

import json
import os
from pathlib import Path

import numpy as np
import rasterio
import multiprocessing
import re
import cv2
from matplotlib import cm

from loguru import logger
from tqdm import tqdm


def run_calibrate(
    outputs_dirpath: str,
    tiles_dirpath: str,
    *,
    resume: bool = True,
    overwrite: bool = False,
    num_cpus: int | None = None,
) -> None:
    """
    Scan all DSM tiles to find global min/max elevation values.

    Args:
        outputs_dirpath: Directory path for output files.
        tiles_dirpath: Directory containing DSM tiles.
        resume: If True, resume from existing `outputs/calibrate/calibrate.json` and skip tiles already processed.
                If False, do not resume; will not overwrite existing `calibrate.json` unless `overwrite=True`.
        overwrite: If True and resume=False, overwrite any existing `calibrate.json` with new results.
        num_cpus: Number of CPUs to use (None means use all available CPUs, 0 disables multiprocessing).
    """
    outputs_path = Path(outputs_dirpath)
    outputs_path.mkdir(parents=True, exist_ok=True)

    tiles_path = Path(tiles_dirpath)

    if not tiles_path.exists():
        logger.error(f"Tiles directory not found: {tiles_path}")
        return

    # Discover all DSM tiles
    logger.info(f"Scanning for DSM tiles in {tiles_path}...")
    tile_files = list(tiles_path.glob("*_DSM.tif"))

    if not tile_files:
        logger.error(f"No DSM tiles found in {tiles_path}")
        return

    logger.info(f"Found {len(tile_files)} DSM tiles")

    # Harmonized startup log
    cpus = num_cpus if num_cpus is not None else os.cpu_count()
    logger.info(f"Starting calibrate: outputs={outputs_path}, tiles={tiles_path}, cpus={cpus}, resume={resume}")

    # Track global min/max
    global_min = float("inf")
    global_max = float("-inf")
    tiles_processed = 0
    tiles_failed = 0

    # Per-tile stats for detailed output
    tile_stats: list[dict] = []

    # Track failed tile filenames for debugging
    failed_tiles: list[str] = []

    # Resume from existing calibrate.json if 'resume' is True to avoid reprocessing tiles
    calibrate_json = outputs_path / "calibrate.json"
    processed_filenames: set[str] = set()
    if resume and calibrate_json.exists():
        try:
            with open(calibrate_json) as f:
                existing = json.load(f)
            existing_stats = existing.get("tile_stats", [])
            for e in existing_stats:
                fname = e.get("filename")
                if fname and e.get("min") is not None and e.get("max") is not None:
                    tile_stats.append(e)
                    processed_filenames.add(fname)
                    tiles_processed += 1
                    try:
                        global_min = min(global_min, float(e["min"]))
                        global_max = max(global_max, float(e["max"]))
                    except Exception:
                        pass
            logger.info(f"Loaded {len(processed_filenames)} existing tile stats from {calibrate_json}")
        except Exception as e:
            logger.warning(f"Failed to load existing calibrate.json: {e}")
    else:
        if calibrate_json.exists() and not resume and not overwrite:
            logger.info(
                "Existing calibrate.json present and resume=False; not overwriting (set overwrite=True to force)"
            )
            return
        if calibrate_json.exists() and not resume and overwrite:
            logger.info("--overwrite specified: existing calibrate.json will be recomputed")

    # Filter out already-processed tiles (if any)
    remaining_tile_files = [tf for tf in tile_files if tf.name not in processed_filenames]
    if not remaining_tile_files:
        logger.info("All tiles already processed, skipping processing step")
    else:
        # Processing using optional multiprocessing
        # num_cpus semantics: None => use all CPUs, 0 => sequential, N => use N cores
        if num_cpus == 0:
            # Sequential processing
            for tile_file in tqdm(remaining_tile_files, desc="Processing tiles"):
                name, tmin, tmax, err = _process_tile_worker(tile_file)
                if err is None:
                    tile_stats.append({"filename": name, "min": tmin, "max": tmax})
                    global_min = min(global_min, tmin)
                    global_max = max(global_max, tmax)
                    tiles_processed += 1
                else:
                    if err == "no_valid_data":
                        logger.warning(f"No valid data in tile {name}")
                    else:
                        logger.warning(f"Failed to process tile {name}: {err}")
                    tiles_failed += 1
                    failed_tiles.append(name)
        else:
            pool_size = num_cpus if num_cpus is not None else os.cpu_count()
            logger.info(f"Running calibration with {pool_size} worker processes")
            try:
                with multiprocessing.Pool(pool_size) as pool:
                    for name, tmin, tmax, err in tqdm(
                        pool.imap_unordered(_process_tile_worker, remaining_tile_files),
                        total=len(remaining_tile_files),
                        desc="Processing tiles",
                    ):
                        if err is None:
                            tile_stats.append({"filename": name, "min": tmin, "max": tmax})
                            global_min = min(global_min, tmin)
                            global_max = max(global_max, tmax)
                            tiles_processed += 1
                        else:
                            if err == "no_valid_data":
                                logger.warning(f"No valid data in tile {name}")
                            else:
                                logger.warning(f"Failed to process tile {name}: {err}")
                            tiles_failed += 1
                            failed_tiles.append(name)
            except KeyboardInterrupt:
                logger.warning("Calibration interrupted by user")
                # Pool will be terminated automatically by context manager on exception

    # Check if we found any valid data
    if tiles_processed == 0:
        logger.error("No tiles were successfully processed")
        return

    # Log results
    logger.info(f"Processed {tiles_processed} tiles ({tiles_failed} failed)")
    logger.info(f"Global min elevation: {global_min}")
    logger.info(f"Global max elevation: {global_max}")

    # Enrich tile_stats with parsed lat/lon and build max elevation image
    def _parse_lat_lon_from_filename(filename: str) -> tuple[int, int] | None:
        pattern = r"([NS])(\d{3})([EW])(\d{3})"
        m = re.search(pattern, filename)
        if not m:
            return None
        lat_char, lat_val, lon_char, lon_val = m.groups()
        lat = int(lat_val) * (-1 if lat_char == "S" else 1)
        lon = int(lon_val) * (-1 if lon_char == "W" else 1)
        return lat, lon

    # Create an image with lat from -90..90 and lon from 0..180 -> size (181 x 181)
    LAT_MIN = -90
    LAT_MAX = 90
    LON_MIN = -180
    LON_MAX = 180
    LAT_STEPS = LAT_MAX - LAT_MIN + 1
    LON_STEPS = LON_MAX - LON_MIN + 1

    # Initialize RGB image (BGR for OpenCV)
    max_map = np.zeros((LAT_STEPS, LON_STEPS, 3), dtype=np.uint8)  # initialized to black

    # Update tile_stats entries to include lat/lon
    updated_tile_stats: list[dict] = []
    for entry in tile_stats:
        filename = entry.get("filename")
        coords = _parse_lat_lon_from_filename(filename)
        if coords is None:
            updated_tile_stats.append({**entry})
            continue
        lat, lon = coords
        entry_with_coords = {**entry, "lat": lat, "lon": lon}
        updated_tile_stats.append(entry_with_coords)

        # Only map tiles with lon in [0,180]
        if entry.get("max") is None:
            continue
        if lon < LON_MIN or lon > LON_MAX:
            continue
        if lat < LAT_MIN or lat > LAT_MAX:
            continue

        # Map to pixel coordinates
        x_idx = lon - LON_MIN  # 0..LON_STEPS-1
        y_idx = LAT_MAX - lat  # lat 90 -> y 0; lat -90 -> y LAT_STEPS-1

        # Normalize elevation value between global_min and global_max
        if global_max > global_min:
            v = (entry["max"] - global_min) / (global_max - global_min)
            v = max(0.0, min(1.0, v))
        else:
            v = 0.0

        # Color mapping: use standard 'terrain' colormap from matplotlib
        rgba = cm.get_cmap("terrain")(v)
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        max_map[y_idx, x_idx] = (b, g, r)

    # Filter out any entries that failed (missing min or max) before saving
    final_tile_stats = [e for e in updated_tile_stats if e.get("min") is not None and e.get("max") is not None]
    excluded_count = len(updated_tile_stats) - len(final_tile_stats)
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} failed/incomplete tile entries from output JSON")

    # Save results
    results = {
        "global_min": global_min,
        "global_max": global_max,
        "tiles_processed": tiles_processed,
        "tiles_failed": tiles_failed,
        "failed_tiles": failed_tiles,
        "tiles_dirpath": str(tiles_path),
        "tile_stats": final_tile_stats,
    }

    json_path = outputs_path / "calibrate.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save map image
    map_path = outputs_path / "max_elevation_map.png"
    try:
        cv2.imwrite(str(map_path), max_map)
        logger.info(f"Max elevation map saved to {map_path}")
    except Exception as e:
        logger.error(f"Failed to save max elevation map: {e}")

    # Create peak-to-peak (max-min) map and save it
    # Gather ptp values
    ptp_values = []
    ptp_entries: list[tuple[int, int, float]] = []  # (lat, lon, ptp)
    for entry in final_tile_stats:
        amin = entry.get("min")
        amax = entry.get("max")
        lat = entry.get("lat")
        lon = entry.get("lon")
        if amin is None or amax is None or lat is None or lon is None:
            continue
        if lon < LON_MIN or lon > LON_MAX:
            continue
        if lat < LAT_MIN or lat > LAT_MAX:
            continue
        ptp = float(amax) - float(amin)
        ptp_values.append(ptp)
        ptp_entries.append((lat, lon, ptp))

    ptp_map = np.zeros((LAT_STEPS, LON_STEPS, 3), dtype=np.uint8)

    if ptp_values:
        ptp_min = min(ptp_values)
        ptp_max = max(ptp_values)
        for lat, lon, ptp in ptp_entries:
            x_idx = int(lon - LON_MIN)
            y_idx = int(LAT_MAX - lat)
            if ptp_max > ptp_min:
                v = (ptp - ptp_min) / (ptp_max - ptp_min)
                v = max(0.0, min(1.0, v))
            else:
                v = 0.0
            rgba = cm.get_cmap("plasma")(v)
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            ptp_map[y_idx, x_idx] = (b, g, r)

        ptp_path = outputs_path / "ptp_elevation_map.png"
        try:
            cv2.imwrite(str(ptp_path), ptp_map)
            logger.info(f"PTP elevation map saved to {ptp_path}")
        except Exception as e:
            logger.error(f"Failed to save PTP elevation map: {e}")

        # Add ptp summary to results
        results["ptp_min"] = float(ptp_min)
        results["ptp_max"] = float(ptp_max)
    else:
        logger.info("No PTP values found; skipping PTP map generation")

    logger.info(f"Calibration results saved to {json_path}")


def _process_tile_worker(tile_file: Path) -> tuple[str, float | None, float | None, str | None]:
    """
    Worker to process a single tile file. Returns tuple:
    (filename, min_value or None, max_value or None, error_message or None)
    """
    try:
        data = None
        with rasterio.open(tile_file) as src:
            data = src.read(1).astype(np.float32)

        valid_data = data[data > -1000]
        if valid_data.size > 0:
            return (tile_file.name, float(valid_data.min()), float(valid_data.max()), None)
        else:
            return (tile_file.name, None, None, "no_valid_data")
    except Exception as e:
        return (tile_file.name, None, None, str(e))
