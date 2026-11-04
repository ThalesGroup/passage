"""Joint operational-design-domain occupancy and descriptive cell summaries.

Cells describe the supplied populations. Counts are not independent samples,
missing metadata is retained, and shared occupancy does not establish causality.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_AXES = (
    "resolution",
    "cost_model",
    "cost_weight",
    "task_variant",
    "terrain_stratum",
    "obstacle_stratum",
    "endpoint_stratum",
)


def _finite_summary(rows: Sequence[Mapping[str, Any]], key: str, statistic: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    if not values:
        return None
    return float(np.mean(values) if statistic == "mean" else np.median(values))


def joint_odd_coverage(
    populations: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    axes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe joint cells under explicit axes, retaining unknown axis values.

    With no explicit axes, use available declared default axes; if none are
    available the status is unknown. Empty populations cannot supply support.
    """
    if not populations:
        raise ValueError("at least one named population is required")
    using_default_axes = axes is None
    if axes is None:
        axes = [axis for axis in DEFAULT_AXES if any(axis in row for rows in populations.values() for row in rows)]
    axes = list(axes)
    if len(set(axes)) != len(axes) or any(not isinstance(axis, str) or not axis for axis in axes):
        raise ValueError("coverage axes must be unique nonempty names")
    names = list(populations)
    missing = {
        name: {axis: sum(row.get(axis) is None for row in rows) for axis in axes} for name, rows in populations.items()
    }
    cells: dict[str, dict[str, Any]] = {}
    for name, rows in populations.items():
        for row in rows:
            coordinates = {axis: row.get(axis) for axis in axes}
            key = json.dumps(coordinates, sort_keys=True, allow_nan=False)
            cell = cells.setdefault(key, {"coordinates": coordinates, "rows": {label: [] for label in names}})
            cell["rows"][name].append(row)
    output = []
    common = 0
    for key in sorted(cells):
        cell = cells[key]
        # Unknown coordinates cannot demonstrate common support.
        known = bool(axes) and all(value is not None for value in cell["coordinates"].values())
        is_common = known and all(cell["rows"][name] for name in names)
        common += int(is_common)
        summaries = {}
        for name in names:
            rows = cell["rows"][name]
            legal_known = [row for row in rows if row.get("legal") is not None]
            summaries[name] = {
                "count": len(rows),
                "legal_known_count": len(legal_known),
                "legal_count": sum(bool(row["legal"]) for row in legal_known),
                "legal_rate": (
                    sum(bool(row["legal"]) for row in legal_known) / len(legal_known) if legal_known else None
                ),
                "mean_cost_regret": _finite_summary(rows, "cost_regret", "mean"),
                "median_cost_regret": _finite_summary(rows, "cost_regret", "median"),
                "median_latency_ms": _finite_summary(rows, "latency_ms", "median"),
                # Censored requests enter that median as recorded lower bounds; count them when rows say so.
                "latency_censored_count": (
                    sum(row.get("latency_censored") is True for row in rows)
                    if any("latency_censored" in row for row in rows)
                    else None
                ),
            }
        output.append({"coordinates": cell["coordinates"], "common_support": is_common, "populations": summaries})
    incomplete = any(count for counts in missing.values() for count in counts.values())
    status = "unknown_axes" if not axes else ("incomplete_metadata" if incomplete else "observed")
    support_status = "unknown" if status != "observed" else ("present" if common else "absent")
    return {
        "status": status,
        "axes": axes,
        "unknown_default_axes": [axis for axis in DEFAULT_AXES if axis not in axes] if using_default_axes else [],
        "population_counts": {name: len(rows) for name, rows in populations.items()},
        "missing_axis_counts": missing,
        "cells": output,
        "n_common_cells": common,
        "common_support_status": support_status,
        "interpretation": "descriptive_only_no_independence_or_causal_claim",
        "limitation": "Occupancy depends on the declared axes and supplied populations; missing axes are unknown.",
    }


def stratify_numeric(values: Sequence[float | None], predeclared_edges: Sequence[float]) -> list[int | None]:
    """Map values to fixed bin indices; intervals are [low, high), last [low, high].

    Missing, nonfinite and outside-range values remain unknown. Boundaries must
    be supplied before inspecting outcomes; the function never estimates edges.
    """
    edges = np.asarray(predeclared_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("predeclared edges must be finite, strictly increasing and contain at least two boundaries")
    bins = []
    for value in values:
        if value is None or not np.isfinite(value) or value < edges[0] or value > edges[-1]:
            bins.append(None)
        else:
            bins.append(min(int(np.searchsorted(edges, value, side="right")) - 1, len(edges) - 2))
    return bins
