"""Prototype checks only; the worker's actual loaded files still need binding."""

from pathlib import Path
from passage_eval.campaign.edge.target import _sha256_regular_file
from passage_eval.io import pinned


def verify_files(paths, expected):
    names = {"bundle", "checkpoint", "engine", "decoder"}
    if set(paths) != names or set(expected) != names:
        raise ValueError("bundle, checkpoint, engine and decoder pins are required")
    actual = {}
    for name in sorted(names):
        actual[name] = _sha256_regular_file(paths[name])
        if actual[name] != expected[name]:
            raise ValueError(f"{name} digest differs")
    data = pinned.strict_json_loads(pinned.read_bounded(Path(paths["bundle"])), name="bundle")
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("lineage"), dict)
        or data["lineage"].get("checkpoint_sha256") != actual["checkpoint"]
    ):
        raise ValueError("bundle checkpoint lineage differs")
    if _sha256_regular_file(paths["bundle"]) != actual["bundle"]:
        raise ValueError("bundle changed while checking lineage")
    return actual
