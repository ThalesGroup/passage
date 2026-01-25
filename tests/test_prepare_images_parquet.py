import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from passage.prepare import _write_parquet_shard


def test_write_parquet_writes_image_and_sets_path(tmp_path: Path):
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

    shard_path = tmp_path / "data-00000.parquet"
    images_dir = tmp_path / "images" / "256" / "train"

    # Run the function under test
    _write_parquet_shard(rows, shard_path, 256, images_dir=images_dir)

    assert shard_path.exists()

    table = pq.read_table(shard_path)
    assert "image" in table.schema.names

    image_field = table.column("image").to_pylist()[0]
    assert isinstance(image_field, dict)
    assert "bytes" in image_field and isinstance(image_field["bytes"], (bytes, bytearray))
    assert "path" in image_field and image_field["path"] is not None

    path_value = image_field["path"]
    # Strip leading slash if present
    if isinstance(path_value, str) and path_value.startswith("/"):
        path_value = path_value[1:]

    file_path = tmp_path / path_value
    assert file_path.exists()
