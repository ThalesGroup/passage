import io

import pytest
from PIL import Image

from passage.export import _rows_to_dataset


@pytest.mark.unit
def test_rows_to_dataset_builds_image_feature():
    # Create a tiny PNG in-memory
    im = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    rows = [
        {
            "image": img_bytes,
            "id": "123",
            "tensor": b"\x00\x01",
            "path_free": [[1, 2], [3, 4]],
            "path_obstacles": [],
            "metadata": {"k": "v"},
        }
    ]

    ds = _rows_to_dataset(rows)

    assert len(ds) == 1
    row = ds[0]
    assert row["id"] == "123"
    assert row["image"] is not None
    assert isinstance(row["tensor"], (bytes, bytearray))
