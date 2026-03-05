import pandas as pd
import pytest

import passage.download as dl


class DummyResp:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self._content = content

    def iter_content(self, chunk_size=8192):
        yield self._content


def test_download_tile_head_exception(monkeypatch):
    # Simulate requests.head raising an exception -> url_valid False
    def head_exc(url, timeout=1):
        raise dl.requests.exceptions.RequestException("network error")

    monkeypatch.setattr(dl.requests, "head", head_exc)
    res = dl.download_tile((0, 0, None))
    assert res[3] is False and res[4] is False


def test_generate_status_map_empty_and_bounds(tmp_path, caplog):
    # Empty DataFrame should be handled and image still created
    df_empty = pd.DataFrame(columns=["lat_min", "lon_min", "url", "url_valid"])
    out = tmp_path / "status.png"
    dl.generate_status_map(df_empty, out)
    assert out.exists()

    # Now test an out-of-bounds entry triggers a warning and still writes image
    df = pd.DataFrame([{"lat_min": 999, "lon_min": 999, "url": "http://x", "url_valid": False}])
    out2 = tmp_path / "status2.png"
    dl.generate_status_map(df, out2)
    assert out2.exists()


@pytest.mark.unit
def test_download_tile_url_not_found(monkeypatch):
    # Simulate head returning 404
    monkeypatch.setattr(dl.requests, "head", lambda url, timeout=1: DummyResp(status_code=404))
    lat, lon, url, url_valid, extracted = dl.download_tile((5, 5, None))
    assert url_valid is False and extracted is False
