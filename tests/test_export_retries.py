import os

from datasets import Dataset
from httpx import ReadTimeout

from passage import export


def test_push_retries_on_transient_readtimeout(monkeypatch, caplog):
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
        [{"id": "1", "image": None, "tensor": b"", "metadata": "{}", "path_free": [], "path_obstacles": []}]
    )
    split_datasets = {"train": ds}

    # Should not raise; should retry and succeed on 3rd attempt
    export._push_resolution(
        api=None,
        token="t",
        repo_id="u/r",
        resolution=64,
        split_datasets=split_datasets,
        private=False,
        max_shard_size="10GB",
    )

    assert calls["n"] == 3
    assert "retrying" in caplog.text or "timed out" in caplog.text
