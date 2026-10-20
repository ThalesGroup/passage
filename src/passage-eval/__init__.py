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

"""passage-eval: canonical evaluation contract for the PASSAGE benchmark.

The public API follows the evaluation protocol of the PASSAGE paper:

    Primary metrics:
        cost_regret, cost_regret_summary,
        path_assd, path_hausdorff_p95,
        success_rate, obstacle_violation_rate,
        compute_primary_metrics, paired_bootstrap_ci

    Statistical discipline:
        DEFAULT_SEEDS, PRIMARY_METRICS, MetricFamily,
        apply_censored_penalty,
        paired_bootstrap_delta, multiseed_paired_bootstrap,
        holm_bonferroni, decide_pairwise,
        BootstrapDelta, HolmBonferroniResult, PairwiseDecision

    Approach-agnostic planner interface:
        Planner, PlannerOutput, Scene, score_planner, evaluate_planner

    Trustworthiness axes:
        TrustworthinessAxis,
        stability_cv, geographic_generalisation,
        odd_boundary_delta, oracle_relative_efficiency

    Per-sample prediction loaders (notebook / reviewer entry-point):
        PerSampleArrays, load_per_sample_npz,
        primary_metrics_from_arrays

    Reference classical planners (illustrative, NOT headline):
        AStarPlanner, GreedyToGoalPlanner, RandomPolicyPlanner

    Decoder-readiness diagnostics:
        mask_endpoints_in_same_component, boundary_f_score,
        descent_direction_agreement, bellman_residual_reachable,
        local_ordinal_concordance, field_ssi_mae

    Inference / deployment footprint:
        benchmark_forward, benchmark_decoder, benchmark_end_to_end,
        astar_expansions_ratio, decoder_recovery_rate

    Reproducibility sidecar:
        ReproducibilitySidecar, compute_calibration_decisions_hash
"""

from passage_eval.metrics.geometric import (
    path_assd,
    path_hausdorff_p95,
    path_iou_diagnostic,
)
from passage_eval.metrics.regret import (
    cost_regret,
    cost_regret_summary,
)
from passage_eval.metrics.reliability import (
    obstacle_violation_rate,
    obstacle_cells_traversed_per_path,
    success_rate,
)
from passage_eval.metrics.summary import (
    compute_primary_metrics,
    paired_bootstrap_ci,
)
from passage_eval.diagnostics.segmentation import (
    boundary_f_score,
    mask_connected_components_total,
    mask_endpoints_in_same_component,
)
from passage_eval.diagnostics.regression import (
    bellman_residual_reachable,
    descent_direction_agreement,
    field_ssi_mae,
    local_ordinal_concordance,
)
from passage_eval.deployment.efficiency import (
    astar_expansions_ratio,
    decoder_recovery_rate,
)
from passage_eval.deployment.throughput import (
    benchmark_decoder,
    benchmark_end_to_end,
    benchmark_forward,
)
from passage_eval.protocol.discipline import (
    DEFAULT_SEEDS,
    PRIMARY_METRICS,
    BootstrapDelta,
    HolmBonferroniResult,
    MetricFamily,
    PairwiseDecision,
    apply_censored_penalty,
    decide_pairwise,
    holm_bonferroni,
    multiseed_paired_bootstrap,
    paired_bootstrap_delta,
)
from passage_eval.protocol.planner import (
    evaluate_planner,
    Planner,
    PlannerOutput,
    Scene,
    score_planner,
)
from passage_eval.io.predictions import (
    PerSampleArrays,
    load_per_sample_npz,
    primary_metrics_from_arrays,
)
from passage_eval.trustworthiness import (
    TrustworthinessAxis,
    geographic_generalisation,
    odd_boundary_delta,
    oracle_relative_efficiency,
    stability_cv,
)
from passage_eval.baselines import (
    AStarPlanner,
    GreedyToGoalPlanner,
    RandomPolicyPlanner,
)
from passage_eval.results import (
    Submission,
    Leaderboard,
    submit,
    load_leaderboard,
    rebuild_leaderboard,
    plot_primary_bars,
    plot_pareto_accuracy_latency,
    plot_trustworthiness_radar,
    plot_pairwise_heatmap,
    plot_grouped_bars_by_accelerator,
    plot_latency_heatmap,
    plot_metric_vs_accelerator,
    summarise_leaderboard,
    scoreboard_markdown,
)
from passage_eval.sidecar import (
    ReproducibilitySidecar,
    compute_calibration_decisions_hash,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # primary metrics
    "cost_regret",
    "cost_regret_summary",
    "path_assd",
    "path_hausdorff_p95",
    "path_iou_diagnostic",
    "success_rate",
    "obstacle_violation_rate",
    "obstacle_cells_traversed_per_path",
    "compute_primary_metrics",
    "paired_bootstrap_ci",
    # statistical discipline
    "DEFAULT_SEEDS",
    "PRIMARY_METRICS",
    "MetricFamily",
    "BootstrapDelta",
    "HolmBonferroniResult",
    "PairwiseDecision",
    "apply_censored_penalty",
    "paired_bootstrap_delta",
    "multiseed_paired_bootstrap",
    "holm_bonferroni",
    "decide_pairwise",
    # planner abstraction (cross-family adapter, App. A.2)
    "Planner",
    "PlannerOutput",
    "Scene",
    "score_planner",
    "evaluate_planner",
    # I/O for per-sample prediction arrays
    "PerSampleArrays",
    "load_per_sample_npz",
    "primary_metrics_from_arrays",
    # trustworthiness axes (§5.4 / App. K)
    "TrustworthinessAxis",
    "stability_cv",
    "geographic_generalisation",
    "odd_boundary_delta",
    "oracle_relative_efficiency",
    # reference baselines (illustrative, not headline)
    "AStarPlanner",
    "GreedyToGoalPlanner",
    "RandomPolicyPlanner",
    # diagnostics — segmentation
    "mask_connected_components_total",
    "mask_endpoints_in_same_component",
    "boundary_f_score",
    # diagnostics — regression
    "descent_direction_agreement",
    "bellman_residual_reachable",
    "local_ordinal_concordance",
    "field_ssi_mae",
    # deployment footprint
    "benchmark_forward",
    "benchmark_decoder",
    "benchmark_end_to_end",
    "astar_expansions_ratio",
    "decoder_recovery_rate",
    # reproducibility sidecar
    # leaderboard storage + visualisations
    "Submission",
    "Leaderboard",
    "submit",
    "load_leaderboard",
    "rebuild_leaderboard",
    "plot_primary_bars",
    "plot_pareto_accuracy_latency",
    "plot_trustworthiness_radar",
    "plot_pairwise_heatmap",
    "plot_grouped_bars_by_accelerator",
    "plot_latency_heatmap",
    "plot_metric_vs_accelerator",
    "summarise_leaderboard",
    "scoreboard_markdown",
    "ReproducibilitySidecar",
    "compute_calibration_decisions_hash",
]
