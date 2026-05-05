# MIT License - Copyright (c) 2026 THALES
"""Deployment-footprint utilities.

Two submodules:
* ``throughput``  : wall-clock benchmarking of the bare neural network
                    forward pass, the canonical decoder, and the
                    end-to-end pipeline (forward + decoder).
* ``efficiency`` : surrogate-acceleration metrics (A*-expansions ratio
                    and decoder-recovery rate).

Reported alongside the primary-metric family but not under the same
Holm--Bonferroni umbrella (cf. App. J Rule 7 of `main.tex`).
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
