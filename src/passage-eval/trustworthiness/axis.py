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

"""Common record type for trustworthiness-axis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrustworthinessAxis:
    """One row of the trustworthiness sub-table.

    ``value`` is the headline scalar (e.g. CV for stability, difference
    of means for geographic / ODD-edge, speed-up factor for efficiency).
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
