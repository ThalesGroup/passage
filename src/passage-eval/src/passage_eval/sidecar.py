# MIT License - Copyright (c) 2026 THALES
"""Reproducibility-sidecar manifest (App. J Rule 8).

Every headline row of the PASSAGE results table carries a sidecar JSON
manifest pinning, at minimum, the artefacts needed to map a number back
to the configuration that produced it:

* dataset version + content hash
* seed set and split seed
* deterministic hash of every calibration-split decision
* reference hardware stack ({"R1", "R2"} or arbitrary string)
* inference batch size and numerical precision
* determinism flag

This module is deliberately small: a single dataclass, deterministic
hashing of arbitrary nested-Python decisions dicts, and JSON
read/write. Submitters are expected to fill the dataclass once per
headline row and ship the JSON with the predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from passage_eval.protocol.discipline import DEFAULT_SEEDS

VALID_DETERMINISM_FLAGS = ("bit_exact", "algorithmically_deterministic", "non_deterministic")
VALID_HARDWARE_STACKS = ("R1", "R2")
VALID_DTYPES = ("fp32", "fp16", "bf16", "tf32", "int8")


def compute_calibration_decisions_hash(decisions: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over a calibration-decisions mapping.

    Renders the mapping as canonical JSON (``sort_keys=True``,
    ``separators=(",", ":")``) before hashing so two manifests with the
    same logical content always produce the same hash. ``Path``,
    ``set``, and ``tuple`` values are normalised for serialisation.
    """
    payload = json.dumps(_normalise_for_json(decisions), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_for_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalise_for_json(x) for x in obj)
    if isinstance(obj, tuple):
        return [_normalise_for_json(x) for x in obj]
    if isinstance(obj, list):
        return [_normalise_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _normalise_for_json(v) for k, v in obj.items()}
    return obj


@dataclass
class ReproducibilitySidecar:
    """Sidecar manifest for one headline row.

    Required minimum fields are taken from App. J §Reproducibility
    Sidecar of the PASSAGE paper. Extra information goes under
    ``extra``; the calibration_decisions_hash should be computed via
    :func:`compute_calibration_decisions_hash` on the original mapping.
    """

    method_name: str
    family: str  # "dense_segmentation" | "dense_regression" | "graph" | "iterative" | ...
    dataset_version: str
    dataset_content_hash: str
    operating_point: str  # e.g. "256x256/elevation"
    seeds: Sequence[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    split_seed: int = 0
    calibration_decisions_hash: str = ""
    hardware_stack: str = "R1"  # "R1" (server H100), "R2" (edge GPU), or free-form
    batch_size: int = 1
    dtype: str = "fp16"
    determinism_flag: str = "algorithmically_deterministic"
    container_digest: str | None = None
    commit_sha: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.determinism_flag not in VALID_DETERMINISM_FLAGS:
            raise ValueError(f"determinism_flag={self.determinism_flag!r} not in {VALID_DETERMINISM_FLAGS}")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"dtype={self.dtype!r} not in {VALID_DTYPES}")
        # hardware_stack is free-form but warn on the canonical labels.
        # (No raise so submitters can label e.g. "R3-jetson-orin".)
        self.seeds = list(int(s) for s in self.seeds)

    def to_dict(self) -> dict[str, Any]:
        return _normalise_for_json(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReproducibilitySidecar":
        # Fan out to the dataclass while preserving ``extra``.
        known = {
            "method_name",
            "family",
            "dataset_version",
            "dataset_content_hash",
            "operating_point",
            "seeds",
            "split_seed",
            "calibration_decisions_hash",
            "hardware_stack",
            "batch_size",
            "dtype",
            "determinism_flag",
            "container_digest",
            "commit_sha",
            "notes",
            "extra",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in payload.items():
            if k in known and k != "extra":
                kwargs[k] = v
            elif k == "extra":
                extra.update(dict(v))
            else:
                extra[k] = v
        kwargs["extra"] = {**kwargs.get("extra", {}), **extra}
        return cls(**kwargs)

    @classmethod
    def read(cls, path: str | Path) -> "ReproducibilitySidecar":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
