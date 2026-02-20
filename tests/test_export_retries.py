import os

from datasets import Dataset
from httpx import ReadTimeout

from passage import export


def test_push_retries_on_transient_readtimeout(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_push_to_hub(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ReadTimeout("simulated read timeout")
        return None

    monkeypatch.setattr(Dataset, "push_to_hub", fake_push_to_hub, raising=False)

    os.environ["HF_UPLOAD_RETRIES"] = "3"
    os.environ["HF_UPLOAD_RETRY_DELAY"] = "0"

    ds = Dataset.from_list(
        [{"id": "1", "tensor": b"", "metadata": "{}", "path_astar_free": [], "path_astar_obstacles": []}]
    )
    split_datasets = {"train": ds}

    # Should not raise; should retry and succeed on 3rd attempt
    export._push_resolution(
        api=None,
        export_dirpath=tmp_path,
        token="t",
        repo_id="u/r",
        resolution=64,
        split_datasets=split_datasets,
        private=False,
        chunk_size=100,
    )

    # Verify retries happened
    assert calls["n"] == 3
