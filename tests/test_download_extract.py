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

import io
import zipfile


import passage.download as dl


class DummyResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self._content = content

    def iter_content(self, chunk_size=8192):
        yield self._content


def make_zip_bytes(filename: str, member_name: str, member_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member_name, member_bytes)
    return buf.getvalue()


def test_download_tile_extracts_dsm(tmp_path, monkeypatch):
    # Create zip bytes with a DSM entry
    content = make_zip_bytes("test.zip", "somepath/N000E000_DSM.tif", b"dsmdata")

    # Monkeypatch requests.head and get
    monkeypatch.setattr(dl.requests, "head", lambda url, timeout=1: DummyResp(status_code=200))
    monkeypatch.setattr(
        dl.requests, "get", lambda url, timeout=180, stream=True: DummyResp(status_code=200, content=content)
    )

    # Run download_tile with raw dir
    raw_dir = tmp_path
    lat, lon, url, url_valid, extracted = dl.download_tile((0, 0, str(raw_dir)))

    assert url_valid is True
    assert extracted is True
    # Check extracted file exists
    out_file = raw_dir / "tiles" / "N000E000_DSM.tif"
    assert out_file.exists()
    assert out_file.read_bytes() == b"dsmdata"


def test_download_tile_no_dsm_in_archive(tmp_path, monkeypatch):
    content = make_zip_bytes("test.zip", "README.txt", b"none")
    monkeypatch.setattr(dl.requests, "head", lambda url, timeout=1: DummyResp(status_code=200))
    monkeypatch.setattr(
        dl.requests, "get", lambda url, timeout=180, stream=True: DummyResp(status_code=200, content=content)
    )

    raw_dir = tmp_path
    lat, lon, url, url_valid, extracted = dl.download_tile((5, 5, str(raw_dir)))

    assert url_valid is True
    assert extracted is False
