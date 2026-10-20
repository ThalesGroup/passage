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

"""Deployment-footprint utilities.

Two submodules:

* ``throughput``: wall-clock benchmarking of the bare neural network
  forward pass, the canonical decoder, and the end-to-end pipeline
  (forward + decoder).
* ``efficiency``: surrogate-acceleration metrics (A*-expansions ratio
  and decoder-recovery rate).

These figures are reported next to the primary metrics but are not part
of the Holm–Bonferroni family.
"""

from passage_eval.deployment.efficiency import (
    astar_expansions_ratio,
    decoder_recovery_rate,
)
from passage_eval.deployment.throughput import (
    benchmark_decoder,
    benchmark_end_to_end,
    benchmark_forward,
)

__all__ = [
    "benchmark_forward",
    "benchmark_decoder",
    "benchmark_end_to_end",
    "astar_expansions_ratio",
    "decoder_recovery_rate",
]
