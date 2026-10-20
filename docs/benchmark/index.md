---
title: Benchmark Overview
icon: material/clipboard-check-outline
---

# Benchmark

**PASSAGE** ships with a self-contained, version-pinned
**evaluation contract** that turns the dataset into a benchmark.
The decoded-path metrics, statistical utilities, trustworthiness diagnostics
and complete-request workflow live under
[`src/passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval).
That package is what you hook up to your planner — whether
it is a dense-vision model, a graph neural network, a
reinforcement-learning policy, or a classical solver — to get a
score with identified inputs, procedure, references and metric contract.
The paper's learned-model results are historical results of the original
study, reported as obtained; not re-run on the released dataset (see
[Paper results](leaderboard.md#paper-results)).

This section of the documentation is the single entry point for that
process. Start with [Evaluate your planner](contributing-a-method.md) to run
`conform → run → score → compare → report`; use the protocol and API pages to
interpret the results.

!!! info "Terms used in the Benchmark pages"

    - **Ordinary input**: what a deployed planner sees for one task: elevation
      in meters, the obstacle mask, the start and goal cells, the graph rules
      and the budget. It never includes the reference route or its cost.
    - **Complete request**: one timed call of the planner, from the ordinary
      input to a route. Input preparation, inference or search, decoding,
      repair and fallback all count, and so do the evaluator's legality check
      and cost recomputation.
    - **Qualified reference**: a stored reference route that is legal, whose
      recomputed cost matches the stored cost, and that an exact solve within
      `reference_max_expansions` confirms as optimal. A run starts only when
      the reference of every task is qualified.
    - **Cohort**: the frozen list of tasks that a result was computed on
      (sample IDs, split, objective and obstacle condition). Two results are
      comparable only on the same cohort.
    - **Censored cost**: the cost given to a route that fails or crosses an
      obstacle, 10⁶ times the reference cost, so that every failure stays in
      the regret statistics.

<div class="grid cards" markdown>

- :material-format-list-numbered: **Evaluation Protocol**

    ---

    The pre-registered, paper-aligned rules: six headline metrics,
    paired-bootstrap CI on the mean of paired differences,
    three-seed pooled aggregation, Holm–Bonferroni step-down
    correction, and the Pareto-dominant / Trade-off / Indistinguishable
    decision rule.

    [:octicons-arrow-right-24: Read the protocol](protocol.md)

- :material-package-variant-closed: **`passage-eval` package**

    ---

    Install, public API, and module-by-module guide for the evaluation
    package. It is separate from the dataset generator, so the evaluation
    rules can be fixed without changing the released data.

    [:octicons-arrow-right-24: Read the package guide](passage-eval.md)

- :material-rocket-launch: **Paired statistics (paper)**

    ---

    The paper's paired mean-difference analysis: pairing checks, three-seed
    aggregation, Holm–Bonferroni correction and sidecar checks. To compare
    planners with the CLI workflow, use
    [Evaluate your planner](contributing-a-method.md).

    [:octicons-arrow-right-24: Run the paired analysis](walkthrough.md)

- :material-trophy-outline: **Leaderboard**

    ---

    Entries scored on PASSAGE v1.0, organized by resolution, objective and
    accelerator. Published entries carry no host or user names. The
    leaderboard has no entries yet; the paper results are listed separately.

    [:octicons-arrow-right-24: Open the leaderboard](leaderboard.md)

- :material-account-plus-outline: **Evaluate your planner**

    ---

    Wrap ordinary inputs, freeze a specification, qualify references, run the
    complete procedure, score saved routes offline and report matched
    comparisons. Includes a runnable external CPU example and component controls.

    [:octicons-arrow-right-24: Evaluate your planner](contributing-a-method.md)

- :material-notebook-multiple: **Reproducible Notebooks**

    ---

    Four notebooks: an evaluator walkthrough, the paper results, the classical
    baselines and the results format. Re-run with `make docs-prepare`.

    [:octicons-arrow-right-24: Open the notebook track](notebooks/index.md)

- :material-api: **API Reference**

    ---

    Auto-generated docstrings for the public symbols of
    `passage_eval.protocol`, `passage_eval.execution`,
    `passage_eval.metrics`, `passage_eval.diagnostics`,
    `passage_eval.trustworthiness`, `passage_eval.deployment` and
    `passage_eval.sidecar`.

    [:octicons-arrow-right-24: Browse the API](api.md)

</div>

---

## Why a separate package?

The dataset and the evaluation contract change at different rates: the
released data is frozen, while the evaluation rules can grow as the community
adds trustworthiness or deployment checks. `passage-eval` therefore lives in
its own Python package, `passage_eval`, but ships in the same `passage`
distribution as the generator: `uv sync --frozen` (or `pip install -e .`) from
the repository root installs both.

You should reach for `passage-eval` when:

- You have a planner and need decoded-path scores on qualified tasks under an
  identified procedure.
- You want to compare planners on identical frozen tasks, with intervals that
  resample whole geographic blocks and with every failure counted. The paper's
  mean-difference and multiple-testing functions are separate from this
  comparison workflow.
- You want a **reproducibility sidecar** for every reported number so
  reviewers can map any score back to the exact dataset version,
  seeds, calibration decisions, and hardware stack that produced it.

You **do not** need `passage-eval` if you only want to *load* the dataset (see
the [Dataset](../dataset/index.md) page) or *generate* samples (see
[Dataset Generation](../guide/index.md)).

---

## Approach scope

The cross-approach evaluation protocol is **planner-family-agnostic**
by design: every primary metric is a
function of a *decoded path* compared against the exact A$^{\star}$
oracle path on the same sample under the same cost model. The
intermediate output of each planner — logits, value field, node
policy, rollout policy — is never scored against the oracle in
isolation.

The paper evaluated two **dense-vision finalists** (path-mask segmentation and
cost-to-go regression). Their numbers are historical results of the original
study, reported as obtained; not re-run on the released dataset. No
learned-model result on PASSAGE v1.0 is published yet; the
[First entries](candidates.md) page describes the first leaderboard entries.
Graph-neural-network and reinforcement-learning planners plug into the same
contract; the paper names them as future work.

---

## Source-data acknowledgement

`passage-eval` is licensed under the project's [MIT
License](https://github.com/ThalesGroup/passage/blob/main/LICENSE). The benchmark itself, however, is built on
the **JAXA ALOS World 3D 30 m DSM** (AW3D30); any user of
**PASSAGE** or its derived artifacts must respect JAXA's
terms of use — see the [JAXA license notice](../index.md#jaxa-license-and-attribution)
on the home page. `DATASET_LICENSE.md` in the repository root summarizes the
terms; the authoritative text is the
[JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/).
