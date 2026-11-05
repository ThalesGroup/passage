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

"""Integrity regressions using small ZIP payloads instead of remote downloads."""

import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest
import requests

import passage.download as dl


class Response:
    status_code = 200

    def __init__(self, payload=b"", interrupted=False):
        self.payload = payload
        self.interrupted = interrupted

    def iter_content(self, chunk_size=8192):
        yield self.payload
        if self.interrupted:
            raise requests.ConnectionError("interrupted stream")

    def close(self):
        pass


def archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        for name, value in entries:
            zipped.writestr(name, value)
    return stream.getvalue()


def serve(monkeypatch, payload, interrupted=False):
    monkeypatch.setattr(dl.requests, "head", lambda *a, **kw: Response())
    monkeypatch.setattr(dl.requests, "get", lambda *a, **kw: Response(payload, interrupted))


def test_completed_archive_records_every_dsm_hash(tmp_path, monkeypatch):
    serve(monkeypatch, archive([("folder/N000E000_DSM.tif", b"first"), ("N001E001_DSM.tif", b"second")]))
    result = dl.download_tile((0, 0, str(tmp_path)))
    assert result[4] is True
    receipt = json.loads((tmp_path / "manifests/N000E000_N005E005.json").read_text())
    assert receipt["status"] == "complete"
    assert {item["filename"] for item in receipt["members"]} == {"N000E000_DSM.tif", "N001E001_DSM.tif"}
    assert receipt["members"][0]["sha256"] == hashlib.sha256(b"first").hexdigest()


def test_bad_second_member_publishes_no_partial_archive(tmp_path, monkeypatch):
    payload = archive([("N000E000_DSM.tif", b"first"), ("N001E001_DSM.tif", b"second")])
    # Corrupt stored member bytes, retaining a parseable central directory.
    payload = payload.replace(b"second", b"damage", 1)
    serve(monkeypatch, payload)
    result = dl.download_tile((0, 0, str(tmp_path)))
    assert result[4] is False
    assert not list((tmp_path / "tiles").glob("*.tif"))


def test_interrupted_download_preserves_existing_tile(tmp_path, monkeypatch):
    (tmp_path / "tiles").mkdir()
    tile = tmp_path / "tiles/N000E000_DSM.tif"
    tile.write_bytes(b"original")
    serve(monkeypatch, archive([("N000E000_DSM.tif", b"replacement")]), interrupted=True)
    assert dl.download_tile((0, 0, str(tmp_path)))[4] is False
    assert tile.read_bytes() == b"original"
    assert not list((tmp_path / "archives").glob("*.part"))


@pytest.mark.parametrize(
    "entries",
    [
        [("../N000E000_DSM.tif", b"first")],
        [("a/N000E000_DSM.tif", b"first"), ("b/N000E000_DSM.tif", b"second")],
        [("N000E000_DSM.tif", b"")],
    ],
)
def test_unsafe_duplicate_or_empty_members_are_rejected(tmp_path, monkeypatch, entries):
    serve(monkeypatch, archive(entries))
    assert dl.download_tile((0, 0, str(tmp_path)))[4] is False
    assert not list((tmp_path / "tiles").glob("*.tif"))


def test_transient_failure_retries_and_publishes_complete_tile(tmp_path, monkeypatch):
    payload = archive([("N000E000_DSM.tif", b"complete")])
    monkeypatch.setattr(dl.requests, "head", lambda *a, **kw: Response())
    responses = iter([Response(b"broken", interrupted=True), Response(payload)])
    monkeypatch.setattr(dl.requests, "get", lambda *a, **kw: next(responses))
    assert dl.download_tile((0, 0, str(tmp_path)))[4] is True
    assert (tmp_path / "tiles/N000E000_DSM.tif").read_bytes() == b"complete"


def test_repeated_failures_are_bounded_and_explicit(tmp_path, monkeypatch):
    serve(monkeypatch, b"invalid zip")
    assert dl.download_tile((0, 0, str(tmp_path)))[4] is False
    receipt = json.loads((tmp_path / "manifests/N000E000_N005E005.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["attempts"] == 3
    assert receipt["error"]


def test_resume_rechecks_missing_hashed_member(tmp_path, monkeypatch):
    serve(monkeypatch, archive([("N000E000_DSM.tif", b"first")]))
    dl.download_tile((0, 0, str(tmp_path)))
    tile = tmp_path / "tiles/N000E000_DSM.tif"
    tile.write_bytes(b"wrong")
    pd.DataFrame(
        [(0, 0, "http://example", True, True)], columns=["lat_min", "lon_min", "url", "url_valid", "extracted"]
    ).to_csv(tmp_path / "download.csv", index=False)
    monkeypatch.setattr(dl, "generate_status_map", lambda *args: None)
    dl.run_download(str(tmp_path), num_cpus=0, archive_pairs=[(0, 0)])
    assert tile.read_bytes() == b"first"
    assert pd.read_csv(tmp_path / "download.csv").iloc[0]["status"] == "complete"


def test_legacy_resume_preserves_tiles_without_claiming_verified_completion(tmp_path, monkeypatch):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles/ALPSMLC30_N000E000_DSM.tif").write_bytes(b"existing")
    pd.DataFrame(
        [(0, 0, "http://example", True, True)], columns=["lat_min", "lon_min", "url", "url_valid", "extracted"]
    ).to_csv(tmp_path / "download.csv", index=False)
    monkeypatch.setattr(
        dl,
        "download_tile",
        lambda *args: (_ for _ in ()).throw(AssertionError("legacy archive must not be redownloaded")),
    )
    monkeypatch.setattr(dl, "generate_status_map", lambda *args: None)
    # Restrict the world grid while preserving actual orchestration/resume.
    monkeypatch.setattr(dl, "range", lambda *args: [0], raising=False)
    dl.run_download(str(tmp_path), num_cpus=0)
    row = pd.read_csv(tmp_path / "download.csv").iloc[0]
    assert row["status"] == "legacy_unverified"
    assert not row["extracted"]
    assert (tmp_path / "tiles/ALPSMLC30_N000E000_DSM.tif").read_bytes() == b"existing"


def test_verified_resume_skips_network_and_deduplicates_csv(tmp_path, monkeypatch):
    serve(monkeypatch, archive([("N000E000_DSM.tif", b"first")]))
    dl.download_tile((0, 0, str(tmp_path)))
    rows = [(0, 0, "http://stale", False, False), (0, 0, "http://stale", True, True)]
    pd.DataFrame(rows, columns=["lat_min", "lon_min", "url", "url_valid", "extracted"]).to_csv(
        tmp_path / "download.csv", index=False
    )
    monkeypatch.setattr(
        dl.requests, "head", lambda *args, **kw: (_ for _ in ()).throw(AssertionError("unexpected network"))
    )
    monkeypatch.setattr(dl, "generate_status_map", lambda *args: None)
    dl.run_download(str(tmp_path), num_cpus=0, archive_pairs=[(0, 0)])
    result = pd.read_csv(tmp_path / "download.csv")
    assert len(result) == 1
    assert result.iloc[0]["status"] == "complete"
    assert json.loads((tmp_path / "download_summary.json").read_text())["all_requested_verified"] is True


def test_unavailable_and_network_failure_have_distinct_receipts(tmp_path, monkeypatch):
    response = Response()
    response.status_code = 404
    monkeypatch.setattr(dl.requests, "head", lambda *args, **kw: response)
    assert dl.download_tile((0, 0, str(tmp_path)))[3:] == (False, False)
    receipt = json.loads((tmp_path / "manifests/N000E000_N005E005.json").read_text())
    assert receipt["status"] == "unavailable" and receipt["attempts"] == 1
    monkeypatch.setattr(dl.requests, "head", lambda *args, **kw: (_ for _ in ()).throw(requests.Timeout("timeout")))
    assert dl.download_tile((0, 0, str(tmp_path)))[3:] == (False, False)
    receipt = json.loads((tmp_path / "manifests/N000E000_N005E005.json").read_text())
    assert receipt["status"] == "failed" and receipt["attempts"] == 3


def test_malformed_resume_does_not_overwrite_existing_csv(tmp_path):
    csv_path = tmp_path / "download.csv"
    csv_path.write_text("lat_min,lon_min\nbroken,0\n")
    with pytest.raises(ValueError, match="malformed"):
        dl.run_download(str(tmp_path), num_cpus=0)
    assert csv_path.read_text() == "lat_min,lon_min\nbroken,0\n"


def test_legacy_csv_is_preserved_before_status_migration(tmp_path, monkeypatch):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles/ALPSMLC30_N000E000_DSM.tif").write_bytes(b"existing")
    csv_path = tmp_path / "download.csv"
    original = "lat_min,lon_min,url,url_valid,extracted\r\n0,0,http://example,True,True\r\n"
    csv_path.write_bytes(original.encode())
    monkeypatch.setattr(dl, "range", lambda *args: [0], raising=False)
    monkeypatch.setattr(dl, "generate_status_map", lambda *args: None)
    dl.run_download(str(tmp_path), num_cpus=0)
    assert (tmp_path / "download.legacy.csv").read_bytes() == original.encode()


def test_selective_resume_marks_unselected_legacy_success_unverified(tmp_path, monkeypatch):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles/ALPSMLC30_N000E000_DSM.tif").write_bytes(b"existing")
    pd.DataFrame(
        [(0, 0, "http://example", True, True)], columns=["lat_min", "lon_min", "url", "url_valid", "extracted"]
    ).to_csv(tmp_path / "download.csv", index=False)
    monkeypatch.setattr(dl, "generate_status_map", lambda *args: None)
    dl.run_download(str(tmp_path), num_cpus=0, archive_pairs=[])
    row = pd.read_csv(tmp_path / "download.csv").iloc[0]
    assert row["status"] == "legacy_unverified"
    assert not row["extracted"]
