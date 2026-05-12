# MIT License
#
# Copyright (c) 2026 THALES
"""I/O helpers for loading PASSAGE per-sample prediction artifacts.

Released models (vision and graph) emit a per-sample NPZ containing the
six primary-metric components (``regret``, ``assd``, ``h95``,
``success``, ``pred_costs``, ``oracle_costs``) plus a few stratifier
columns (``obs_density``, ``elev_std``, ``oracle_len``). This module
turns those NPZ files into the array shape expected by the rest of the
``passage_eval`` API without re-running the model.
"""

from passage_eval.io.predictions import (
    PerSampleArrays,
    load_per_sample_npz,
    primary_metrics_from_arrays,
)

__all__ = [
    "PerSampleArrays",
    "load_per_sample_npz",
    "primary_metrics_from_arrays",
]
