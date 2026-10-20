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

"""
Download module for PASSAGE.

This module downloads ALOS AW3D30 tiles by checking URL availability,
tracks the download status in a CSV file, and generates a PNG status map.
"""

import csv
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

HTTP_STATUS_OK = 200
MAX_ATTEMPTS = 3


def _archive_url(lat: int, lon: int) -> str:
    def coordinate(latitude: int, longitude: int) -> str:
        return f"{'S' if latitude < 0 else 'N'}{abs(latitude):03d}{'W' if longitude < 0 else 'E'}{abs(longitude):03d}"

    filename = f"{coordinate(lat, lon)}_{coordinate(lat + 5, lon + 5)}.zip"
    return f"https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2404/{filename}"


def _manifest_path(raw_dir: Path, url: str) -> Path:
    return raw_dir / "manifests" / (Path(url).stem + ".json")


def _atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(raw_dir: Path, url: str) -> dict:
    try:
        record = json.loads(_manifest_path(raw_dir, url).read_text())
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def _verified_manifest(raw_dir: Path, url: str, record: dict) -> bool:
    if record.get("status") != "complete" or record.get("url") != url or not record.get("members"):
        return False
    try:
        seen = set()
        for member in record["members"]:
            name = member["filename"]
            if Path(name).name != name or name in seen or not name.upper().endswith("_DSM.TIF"):
                return False
            seen.add(name)
            path = raw_dir / "tiles" / name
            if path.is_symlink() or path.stat().st_size != member["size_bytes"] or _sha256(path) != member["sha256"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _extract_archive(zip_path: Path, raw_dir: Path, record: dict) -> None:
    """Validate all DSM members in staging before any final tile is replaced."""
    files_dir = raw_dir / "tiles"
    files_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=raw_dir) as staging:
        members = []
        seen = set()
        with zipfile.ZipFile(zip_path) as archive:
            for entry in archive.infolist():
                if not entry.filename.upper().endswith("_DSM.TIF"):
                    continue
                member_path = PurePosixPath(entry.filename)
                name = member_path.name
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or "\\" in entry.filename
                    or name in seen
                    or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.file_size <= 0
                ):
                    raise ValueError(f"Unsafe, duplicate or empty DSM member: {entry.filename}")
                seen.add(name)
                staged = Path(staging) / name
                with archive.open(entry) as source, staged.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                if staged.stat().st_size != entry.file_size:
                    raise ValueError(f"Incomplete DSM member: {entry.filename}")
                members.append({"filename": name, "size_bytes": entry.file_size, "sha256": _sha256(staged)})
        if not members:
            raise ValueError("Archive contains no DSM members")
        # A crash during publication leaves the receipt incomplete; resume rechecks it.
        record.update(status="publishing", members=members)
        _atomic_json(_manifest_path(raw_dir, record["url"]), record)
        for member in members:
            os.replace(Path(staging) / member["filename"], files_dir / member["filename"])
        record.update(status="complete", members=members, error="")
        _atomic_json(_manifest_path(raw_dir, record["url"]), record)


def download_tile(args: tuple[int, int, str | None]) -> tuple[int, int, str, bool, bool]:
    """Download an archive with bounded retries and a hashed completion receipt.

    ``extracted`` means every DSM member passed CRC/size verification and was
    published atomically as an individual file. Failures and HTTP absence remain
    distinct in the archive receipt; the historical five-field return is kept.
    """
    lat, lon, raw_dir_str = args
    url = _archive_url(lat, lon)
    raw_dir = Path(raw_dir_str) if raw_dir_str is not None else None
    record = {
        "schema_version": 1,
        "lat_min": lat,
        "lon_min": lon,
        "url": url,
        "status": "failed",
        "members": [],
        "error": "",
        "attempts": 0,
    }
    url_valid = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record["attempts"] = attempt
        try:
            response = requests.head(url, timeout=15)
            try:
                status_code = response.status_code
            finally:
                if hasattr(response, "close"):
                    response.close()
            if status_code in (404, 410):
                record.update(status="unavailable", error=f"HEAD HTTP {status_code}")
                break
            if status_code != HTTP_STATUS_OK:
                raise requests.HTTPError(f"HEAD HTTP {status_code}")
            url_valid = True
            if raw_dir is None:
                return lat, lon, url, True, False
            archives = raw_dir / "archives"
            archives.mkdir(parents=True, exist_ok=True)
            zip_path = archives / Path(url).name
            part_path = archives / (Path(url).name + ".part")
            try:
                response = requests.get(url, timeout=180, stream=True)
                try:
                    if response.status_code != HTTP_STATUS_OK:
                        raise requests.HTTPError(f"GET HTTP {response.status_code}")
                    with part_path.open("wb") as destination:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                destination.write(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                finally:
                    if hasattr(response, "close"):
                        response.close()
                os.replace(part_path, zip_path)
                _extract_archive(zip_path, raw_dir, record)
                zip_path.unlink(missing_ok=True)
                return lat, lon, url, True, True
            finally:
                part_path.unlink(missing_ok=True)
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile, EOFError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            logger.warning(f"Archive {Path(url).name} attempt {attempt}/{MAX_ATTEMPTS}: {error}")
    if raw_dir is not None:
        _atomic_json(_manifest_path(raw_dir, url), record)
    return lat, lon, url, url_valid, False


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


def _write_results(csv_path: Path, results: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=csv_path.parent, newline="", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(
                stream, fieldnames=["lat_min", "lon_min", "url", "url_valid", "extracted", "status", "error"]
            )
            writer.writeheader()
            writer.writerows(results[pair] for pair in sorted(results))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _legacy_archive_pairs(raw_dir: Path) -> set[tuple[int, int]]:
    pairs = set()
    for path in (raw_dir / "tiles").glob("*_DSM.tif"):
        match = re.search(r"([NS])(\d{3})([EW])(\d{3})_DSM\.tif$", path.name)
        if match and not path.is_symlink() and path.stat().st_size > 0:
            lat = int(match[2]) * (-1 if match[1] == "S" else 1)
            lon = int(match[4]) * (-1 if match[3] == "W" else 1)
            pairs.add((lat // 5 * 5, lon // 5 * 5))
    return pairs


def run_download(
    outputs_dirpath: str,
    *,
    resume: bool = True,
    num_cpus: int | None = None,
    archive_pairs: list[tuple[int, int]] | None = None,
) -> None:
    """Download the world grid or explicitly refresh selected 5-degree archives.

    Resume validates hashed receipts. Legacy CSV successes remain
    ``legacy_unverified`` without being redownloaded automatically. Explicit
    ``archive_pairs`` retries only those archives, preserving other records
    and tiles. Use ``resume=False`` to force refresh even verified archives.
    Unverified or failed records never count as complete coverage.
    """
    outputs_path = Path(outputs_dirpath)
    outputs_path.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_path / "download.csv"
    results = {}
    legacy_csv = False
    if csv_path.exists() and (resume or archive_pairs is not None):
        try:
            with csv_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                legacy_csv = "status" not in (reader.fieldnames or [])
                for row in reader:
                    pair = int(row["lat_min"]), int(row["lon_min"])
                    row = {
                        key: row.get(key, "")
                        for key in ["lat_min", "lon_min", "url", "url_valid", "extracted", "status", "error"]
                    }
                    row.update(
                        lat_min=pair[0],
                        lon_min=pair[1],
                        url_valid=_csv_bool(row["url_valid"]),
                        extracted=_csv_bool(row["extracted"]),
                    )
                    results[pair] = row
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Cannot safely resume malformed download CSV: {csv_path}") from error

    all_pairs = [(lat, lon) for lat in range(-90, 90, 5) for lon in range(-180, 180, 5)]
    if archive_pairs is not None:
        if any(pair not in all_pairs for pair in archive_pairs):
            raise ValueError("Archive coordinates must lie on the world 5-degree grid")
        selected = sorted(set(archive_pairs))
    else:
        selected = all_pairs
    if legacy_csv and not (outputs_path / "download.legacy.csv").exists():
        with tempfile.NamedTemporaryFile(dir=outputs_path, delete=False) as backup:
            temporary = Path(backup.name)
            try:
                with csv_path.open("rb") as source:
                    shutil.copyfileobj(source, backup)
                backup.flush()
                os.fsync(backup.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, outputs_path / "download.legacy.csv")
        finally:
            temporary.unlink(missing_ok=True)
    legacy_pairs = _legacy_archive_pairs(outputs_path)
    for pair, row in results.items():
        if not row["status"]:
            if row["extracted"]:
                row.update(
                    extracted=False,
                    status="legacy_unverified" if pair in legacy_pairs else "legacy_missing",
                    error="No archive member/hash receipt; legacy CSV success is unverified",
                )
            else:
                row.update(status="legacy_unavailable", error="Historical HEAD failure/absence was not distinguished")
    pending = []
    for pair in selected:
        row = results.get(pair)
        url = _archive_url(*pair)
        record = _read_manifest(outputs_path, url)
        verified = _verified_manifest(outputs_path, url, record) if resume else False
        if verified:
            results[pair] = dict(
                lat_min=pair[0], lon_min=pair[1], url=url, url_valid=True, extracted=True, status="complete", error=""
            )
        elif (
            archive_pairs is None
            and resume
            and row
            and pair in legacy_pairs
            and (row["extracted"] or row["status"] == "legacy_unverified")
            and not record
        ):
            row.update(
                extracted=False,
                status="legacy_unverified",
                error="No archive member/hash receipt; legacy CSV success is unverified",
            )
        else:
            pending.append((*pair, str(outputs_path)))

    _write_results(csv_path, results)

    def record_result(result: tuple[int, int, str, bool, bool]) -> None:
        lat, lon, url, valid, extracted = result
        receipt = _read_manifest(outputs_path, url)
        # Only the on-disk receipt can establish successful extraction.
        complete = extracted and receipt.get("status") == "complete"
        results[(lat, lon)] = dict(
            lat_min=lat,
            lon_min=lon,
            url=url,
            url_valid=valid,
            extracted=complete,
            status=receipt.get("status", "failed"),
            error=receipt.get("error", "Missing archive receipt"),
        )
        _write_results(csv_path, results)

    logger.info(f"Processing {len(pending)} archives; retaining {len(results)} previous records")
    if num_cpus == 0:
        for result in tqdm(map(download_tile, pending), total=len(pending)):
            record_result(result)
    elif pending:
        with multiprocessing.Pool(num_cpus if num_cpus is not None else os.cpu_count()) as pool:
            for result in tqdm(pool.imap(download_tile, pending), total=len(pending)):
                record_result(result)
    df_results = pd.DataFrame(
        list(results.values()), columns=["lat_min", "lon_min", "url", "url_valid", "extracted", "status", "error"]
    )
    generate_status_map(df_results, outputs_path / "world_coverage.png")
    counts = df_results["status"].value_counts().to_dict()
    _atomic_json(
        outputs_path / "download_summary.json",
        {
            "schema_version": 1,
            "scope": "selected_archives" if archive_pairs is not None else "world_grid",
            "requested_archives": len(selected),
            "statuses": counts,
            "all_requested_verified": all(results.get(pair, {}).get("status") == "complete" for pair in selected),
        },
    )
    logger.info(f"Download pass finished: {counts}. Only 'complete' archives have verified member receipts.")
