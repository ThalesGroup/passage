"""Immutable harness pins for the one pre-G3 validation rehearsal.

The operator supplies file locations; the harness retains the fixed provenance
hashes. Missing configuration fails closed and no roster or dataset is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

CONTROL_SCHEMA = "passage-edge-validation-control/v1"


@dataclass(frozen=True)
class ValidationRelayControl:
    """The immutable validation roster and dataset pins owned by the harness."""

    roster_path: Path
    roster_file_sha256: str
    roster_sha256: str
    dataset_root: Path
    pinned_receipt_sha256: str
    schema: str = CONTROL_SCHEMA


_CONTROL: ValidationRelayControl | None = None


def validation_relay_control() -> ValidationRelayControl | None:
    """Return fixed provenance with operator-supplied paths, or None if unset."""
    if _CONTROL is not None:
        return _CONTROL
    roster = os.environ.get("PASSAGE_VALIDATION_ROSTER")
    dataset = os.environ.get("PASSAGE_VALIDATION_DATASET_ROOT")
    if not roster or not dataset:
        return None
    roster_path, dataset_root = Path(roster), Path(dataset)
    if not roster_path.is_absolute() or not dataset_root.is_absolute():
        raise ValueError("validation roster and dataset paths must be absolute")
    return ValidationRelayControl(
        roster_path=roster_path,
        roster_file_sha256="0f29ab361c2313038a5c17f736d57245b1d067b0569f90b5f2a13c568f94cb49",
        roster_sha256="4a4869a18dd7e5ec099149069c74321a11cdf40450f95bac81d74e8fe054b556",
        dataset_root=dataset_root,
        pinned_receipt_sha256="78c745d0c51d36d34da99705e0389b3cf4fe717802ffdee7d208aa2285110ed4",
    )


__all__ = ["validation_relay_control"]
