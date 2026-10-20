# MIT License
#
# Copyright (c) 2026 THALES

"""Common record type for trustworthiness-axis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrustworthinessAxis:
    """One row of the trustworthiness sub-table.

    ``value`` is the headline scalar (e.g. CV for stability, paired
    delta for geographic / ODD-edge, speed-up factor for efficiency).
    ``flag`` is a Boolean trip indicator: ``True`` means the axis
    *fires* (the planner is unstable, generalises poorly, collapses at
    the boundary, or is slower than the oracle). Threshold semantics
    are axis-specific and documented inside each module.
    """

    name: str  #: short axis identifier ("Stab", "Geo", "ODDe", "Eff")
    metric: str  #: which primary metric this axis re-applies
    value: float  #: headline scalar value
    flag: bool = False  #: True if the axis trips its threshold
    n_samples: int = 0  #: sample-count backing the value
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.name,
            "metric": self.metric,
            "value": float(self.value),
            "flag": bool(self.flag),
            "n_samples": int(self.n_samples),
            **{f"extra_{k}": v for k, v in self.extras.items()},
        }
