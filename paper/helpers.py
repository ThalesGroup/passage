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

"""Shared helpers for the PASSAGE dataset-side paper workspace.

This module is intentionally thin. It standardises the handoff contract between
the four paper notebooks under ``passage/paper/notebooks/`` and the figure
artifacts written under ``passage/paper/figures/``. It does not re-implement
functionality that already lives in ``passage/src/passage/``; instead, it
wraps the existing surfaces so each notebook stays short and executable
top-to-bottom, even on a machine without the full dataset share mounted.

Public surface (see ``__all__`` below):

- ``load_tile(tile_id, resolution)`` — thin wrapper around the dataset loader.
- ``run_astar_bench(backend, grid, start, goal, cost_model, weight, heuristic)``
  — single timed A* call for solver-runtime scaling studies.
- ``emit_manifest(study_id, inputs, outputs)`` — writes
  ``passage/paper/figures/<study_id>.manifest.json`` with provenance.
- ``save_figure(fig, study_id)`` — saves the figure as both PDF and PNG under
  ``passage/paper/figures/`` and returns the PDF path.

Paths are resolved relative to this file so the module works whether called
from a notebook, a script, or pytest. The helpers fail with clear messages if
numpy or matplotlib are unavailable, and gracefully degrade (``None`` returns,
warning logs) when the real dataset share is absent — notebooks should treat a
``None`` from ``load_tile`` as a signal to skip quantitative cells.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PAPER_ROOT",
    "FIGURES_DIR",
    "NOTEBOOKS_DIR",
    "BenchResult",
    "load_tile",
    "run_astar_bench",
    "emit_manifest",
    "save_figure",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PAPER_ROOT = Path(__file__).resolve().parent
"""Absolute path to ``passage/paper/``."""

FIGURES_DIR = PAPER_ROOT / "figures"
"""Absolute path to ``passage/paper/figures/``; all manifests and figures land here."""

NOTEBOOKS_DIR = PAPER_ROOT / "notebooks"
"""Absolute path to ``passage/paper/notebooks/``."""


# ---------------------------------------------------------------------------
# Dependency import guards
# ---------------------------------------------------------------------------


def _require_numpy():
    try:
        import numpy as np  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "numpy is required for passage.paper.helpers; "
            "install the passage dev extras (pip install -e '.[dev]')."
        ) from exc
    return np  # type: ignore[name-defined]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required to save figures from passage.paper.helpers; "
            "install the passage dev extras (pip install -e '.[dev]')."
        ) from exc
    return plt  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_tile(tile_id: str, resolution: int):
    """Load a single PASSAGE tile at the requested resolution.

    This is a thin wrapper. The real loader lives under
    ``passage/src/passage/`` (see ``passage.generate`` /
    ``passage.export``). If the dataset share is not reachable, the function
    returns ``None`` and logs a warning so notebooks can skip gracefully.

    Parameters
    ----------
    tile_id:
        Tile identifier, e.g. ``"N046E006"`` for a 1-degree ALOS tile.
    resolution:
        Target resolution in pixels (one of the seven documented in
        Section 3.3 of ``neurips/skeleton.md``).

    Returns
    -------
    np.ndarray | None
        A ``(resolution, resolution, 3)`` ``float32`` array with
        ``(elevation, markers, obstacles)`` channels, or ``None`` if the
        real loader / data share is not available in this environment.
    """
    np = _require_numpy()
    try:
        # Import lazily; keeps the notebook importable on CI runners without
        # the full passage package installed.
        from passage import export as _export  # type: ignore
    except ImportError:
        logger.warning(
            "passage package not importable; load_tile returning None. "
            "See passage/src/passage/ for the real loader."
        )
        return None

    loader = getattr(_export, "load_sample_tile", None)
    if loader is None:
        logger.warning(
            "passage.export.load_sample_tile not found; "
            "update passage/paper/helpers.py::load_tile to match the current loader."
        )
        return None

    try:
        tile = loader(tile_id=tile_id, resolution=resolution)
    except Exception as exc:  # pragma: no cover - I/O
        logger.warning("load_tile(%s, %s) failed: %s", tile_id, resolution, exc)
        return None

    arr = np.asarray(tile, dtype=np.float32)
    return arr


# ---------------------------------------------------------------------------
# Solver benchmarking
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Result payload from a single timed A* invocation."""

    backend: str
    cost_model: str
    heuristic: str
    resolution: int
    wall_clock_s: float
    path_length_cells: int
    path_cost: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cost_model": self.cost_model,
            "heuristic": self.heuristic,
            "resolution": self.resolution,
            "wall_clock_s": self.wall_clock_s,
            "path_length_cells": self.path_length_cells,
            "path_cost": self.path_cost,
            "status": self.status,
        }


def run_astar_bench(
    backend: str,
    grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    cost_model: str = "distance",
    weight: float = 0.0,
    heuristic: str = "euclidean",
) -> BenchResult:
    """Time a single A* call on ``grid`` with the named backend.

    Parameters
    ----------
    backend:
        One of ``"networkx"``, ``"pathfinding"``, ``"passage_grid"``. These
        names match the three backends documented in Section 5.1 of
        ``neurips/skeleton.md``.
    grid:
        ``np.ndarray`` of shape ``(H, W)`` with elevation values; obstacle
        cells are expected to be encoded as ``+inf`` or as a separate mask
        handled by the backend.
    start, goal:
        ``(row, col)`` coordinates.
    cost_model:
        One of ``distance``, ``elevation``, ``absolute_elevation``,
        ``slope_uphill``, ``energy`` (Table 4).
    weight:
        Per-sample cost-model weight.
    heuristic:
        One of ``euclidean``, ``manhattan``, ``chebyshev``.

    Returns
    -------
    BenchResult
        Wall-clock time and path summary. ``status="skipped"`` if the
        requested backend is unavailable in this environment.
    """
    np = _require_numpy()
    grid = np.asarray(grid)
    resolution = int(grid.shape[0])

    try:
        from passage import pathfinding_utils as _pf  # type: ignore
    except ImportError:
        logger.warning(
            "passage.pathfinding_utils not importable; skipping bench call."
        )
        return BenchResult(
            backend=backend,
            cost_model=cost_model,
            heuristic=heuristic,
            resolution=resolution,
            wall_clock_s=float("nan"),
            path_length_cells=0,
            path_cost=float("nan"),
            status="skipped_import",
        )

    solver_fn = getattr(_pf, f"run_astar_{backend}", None)
    if solver_fn is None:
        solver_fn = getattr(_pf, "run_astar", None)

    if solver_fn is None:
        logger.warning(
            "No A* entry point found on passage.pathfinding_utils; "
            "update run_astar_bench to wire this backend (%s).",
            backend,
        )
        return BenchResult(
            backend=backend,
            cost_model=cost_model,
            heuristic=heuristic,
            resolution=resolution,
            wall_clock_s=float("nan"),
            path_length_cells=0,
            path_cost=float("nan"),
            status="skipped_entrypoint",
        )

    t0 = time.perf_counter()
    try:
        path, cost = solver_fn(
            grid=grid,
            start=start,
            goal=goal,
            cost_model=cost_model,
            weight=weight,
            heuristic=heuristic,
        )
        status = "ok"
    except TypeError:
        # Backend signature differs; retry with positional args.
        try:
            path, cost = solver_fn(grid, start, goal)
            status = "ok_positional"
        except Exception as exc:  # pragma: no cover - runtime path
            logger.warning("Bench call failed: %s", exc)
            path, cost, status = [], float("nan"), "error"
    except Exception as exc:  # pragma: no cover - runtime path
        logger.warning("Bench call failed: %s", exc)
        path, cost, status = [], float("nan"), "error"
    wall = time.perf_counter() - t0

    length = len(path) if path is not None else 0
    cost_value = float(cost) if cost is not None else float("nan")

    return BenchResult(
        backend=backend,
        cost_model=cost_model,
        heuristic=heuristic,
        resolution=resolution,
        wall_clock_s=float(wall),
        path_length_cells=int(length),
        path_cost=cost_value,
        status=status,
    )


# ---------------------------------------------------------------------------
# Manifests and figures
# ---------------------------------------------------------------------------


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def emit_manifest(study_id: str, inputs: dict, outputs: list) -> Path:
    """Write ``<study_id>.manifest.json`` under ``passage/paper/figures/``.

    The manifest is a small JSON document that records every input fixed by
    the notebook and every output artifact produced, plus a minimal runtime
    footprint. It is the dataset-side equivalent of the CSVs that the
    ``passage-vision`` workspace emits under ``outputs/paper/results/``.

    Parameters
    ----------
    study_id:
        Short stable identifier, e.g. ``"solver_runtime_scaling"``.
    inputs:
        Arbitrary JSON-serialisable dict describing the study inputs
        (resolutions, cost models, backends, seeds, tile ids, ...).
    outputs:
        Iterable of ``Path`` objects pointing at artifacts the notebook
        produced. Relative paths are resolved against ``FIGURES_DIR``.

    Returns
    -------
    Path
        Absolute path to the written manifest file.
    """
    figures_dir = _ensure_figures_dir()
    manifest_path = figures_dir / f"{study_id}.manifest.json"

    resolved_outputs: list[str] = []
    for item in outputs:
        p = Path(item)
        if not p.is_absolute():
            p = figures_dir / p
        resolved_outputs.append(str(p))

    payload = {
        "study_id": study_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inputs": inputs,
        "outputs": resolved_outputs,
    }

    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return manifest_path


def save_figure(fig, study_id: str) -> Path:
    """Save ``fig`` as both PDF and PNG under ``passage/paper/figures/``.

    Parameters
    ----------
    fig:
        A matplotlib ``Figure`` instance.
    study_id:
        Filename stem, e.g. ``"solver_runtime_scaling"``.

    Returns
    -------
    Path
        Absolute path to the PDF artifact. The PNG sibling is written next
        to it with the same stem.
    """
    _require_matplotlib()
    figures_dir = _ensure_figures_dir()
    pdf_path = figures_dir / f"{study_id}.pdf"
    png_path = figures_dir / f"{study_id}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    return pdf_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke-test the module on import-only surfaces: paths resolve and a
    # trivial manifest round-trips. No data is loaded, no figures produced.
    print("PAPER_ROOT =", PAPER_ROOT)
    print("FIGURES_DIR =", FIGURES_DIR)
    manifest = emit_manifest(
        study_id="_smoke_test",
        inputs={"note": "smoke-test from helpers.py __main__"},
        outputs=[],
    )
    print("wrote manifest:", manifest)
