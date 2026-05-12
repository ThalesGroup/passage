---
title: Benchmark Overview
icon: material/clipboard-check-outline
---

# Benchmark

**PASSAGE** is shipped with a self-contained, version-pinned
**evaluation contract** that turns the dataset into a benchmark.
Every primary metric, every statistical-discipline rule, every
trustworthiness diagnostic, and every inference-footprint utility used
in the **PASSAGE** paper is implemented as a Python module under
[`src/passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval).
That sub-package is what you should hook up to your planner — whether
it is a dense-vision model, a graph-neural network, a
reinforcement-learning policy, or a classical solver — to get a
**PASSAGE**-comparable score.

This section of the documentation is the single entry point for that
process. It covers four orthogonal topics:

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

    Architecture, install, public API, and module-by-module reference
    for the evaluation sub-package. Decoupled from the dataset
    generator so the contract can be patched without invalidating
    released artefacts.

    [:octicons-arrow-right-24: Read the package guide](passage-eval.md)

- :material-rocket-launch: **End-to-end walkthrough**

    ---

    A full worked example: from a trained planner's predictions to a
    paper-comparable headline row, including the censored-penalty
    rule, three-seed aggregation, the Holm–Bonferroni step, and the
    reproducibility-sidecar emission.

    [:octicons-arrow-right-24: Run a full evaluation](walkthrough.md)

- :material-trophy-outline: **Leaderboard**

    ---

    Live snapshot of every submitted run, organised by resolution,
    cost-model, host, and accelerator. Includes the headline
    Pareto frontier, latency heatmap, and per-method bar charts.

    [:octicons-arrow-right-24: Open the leaderboard](leaderboard.md)

- :material-account-plus-outline: **Contributing a method**

    ---

    Step-by-step recipe for hooking up *your* planner — vision, GNN,
    RL, or classical — to the `passage-eval bench` CLI. One YAML spec
    in, one leaderboard row out.

    [:octicons-arrow-right-24: Submit a new method](contributing-a-method.md)

- :material-notebook-multiple: **Reproducible Notebooks**

    ---

    Three executed-on-build notebooks: the cross-family `Planner` ABC
    walkthrough, scoring of the released baselines, and the leaderboard
    visualisation. Re-run with `make docs-prepare`.

    [:octicons-arrow-right-24: Open the notebook track](notebooks/index.md)

- :material-api: **API Reference**

    ---

    Auto-generated docstrings for every public symbol in
    `passage_eval.metrics`, `passage_eval.protocol`,
    `passage_eval.diagnostics`, `passage_eval.deployment`, and
    `passage_eval.sidecar`.

    [:octicons-arrow-right-24: Browse the API](api.md)

</div>

---

## Why a separate package?

The dataset and the evaluation contract evolve in lock-step but at
different cadences — released data tiles are mostly frozen, but
evaluation rules can iterate as the community surfaces new
trustworthiness axes or deployment constraints. Keeping `passage-eval`
as a **sibling sub-package** under the same repository makes the
lock-step explicit while letting you `pip install` either side
independently.

You should reach for `passage-eval` when:

- You have predictions from a planner and need a **score that the
  PASSAGE paper would accept**.
- You want to compare two planners on the same test split and need
  the **paired-bootstrap CI on the mean of paired differences** plus
  family-wise multiple-testing correction.
- You want a **reproducibility sidecar** for every reported number so
  reviewers can map any score back to the exact dataset version,
  seeds, calibration decisions, and hardware stack that produced it.

You **do not** need `passage-eval` if you only want to *generate*
samples or *load* the dataset; for that the
[Dataset Generation](../generation/index.md) section is what you want.

---

## Approach scope

The cross-approach evaluation protocol is **planner-family-agnostic**
by design (Rule 1 in the paper's App. A): every primary metric is a
function of a *decoded path* compared against the exact A$^{\star}$
oracle path on the same sample under the same cost model. The
intermediate output of each planner — logits, value field, node
policy, rollout policy — is never scored against the oracle in
isolation.

The current release exercises the protocol with two **dense-vision
example baselines** (a path-mask segmentation finalist and a
cost-to-go regression finalist). Graph-neural-network and
reinforcement-learning entries plug into the same contract and are
explicitly named in the paper Conclusion and in App. A as
**future-work targets**; they are not part of the present numerical
evidence base.

---

## Source-data acknowledgement

`passage-eval` is licensed under the project's [MIT
License](https://github.com/ThalesGroup/passage/blob/main/LICENSE). The benchmark itself, however, is built on
the **JAXA ALOS World 3D 30 m DSM** (AW3D30); any user of
**PASSAGE** or its derived artefacts must respect JAXA's
licensing terms — see the [JAXA license notice](../index.md#jaxa-license-and-attribution)
on the home page and `DATASET_LICENSE.md` in the repository root for
the full text mirrored from the JAXA terms-of-use page.
