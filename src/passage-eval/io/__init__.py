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
