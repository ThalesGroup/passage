---
title: Benchmark Overview
icon: material/clipboard-check-outline
---

# Benchmark

**PASSAGE** is shipped with a self-contained, version-pinned
**evaluation contract** that turns the dataset into a benchmark.
The decoded-path metrics, statistical utilities, trustworthiness diagnostics
and complete-request workflow live under
[`src/passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval).
That sub-package is what you should hook up to your planner — whether
it is a dense-vision model, a graph-neural network, a
reinforcement-learning policy, or a classical solver — to get a
score with identified inputs, procedure, references and metric contract.
Historical paper values require their own authenticated producer artifacts.

This section of the documentation is the single entry point for that
process. Start with the existing-algorithm guide to run
`conform → run → score → compare → report`; use the protocol and API pages to
interpret the results.

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

    The historical statistical API's worked mean-difference example:
    pairing and failure coverage, three-seed aggregation, Holm–Bonferroni
    correction and sidecar evidence. Current complete-request comparison
    follows the existing-algorithm guide.

    [:octicons-arrow-right-24: Run a full evaluation](walkthrough.md)

- :material-trophy-outline: **Leaderboard**

    ---

    Entries scored on PASSAGE v1.0, organised by resolution, cost model, host
    and accelerator. The leaderboard has no entries yet; the paper results are
    listed separately.

    [:octicons-arrow-right-24: Open the leaderboard](leaderboard.md)

- :material-account-plus-outline: **Harness an existing algorithm**

    ---

    Wrap ordinary inputs, freeze a specification, qualify references, run the
    complete procedure, score persisted routes offline and report matched
    comparisons. Includes a runnable external CPU fixture and component controls.

    [:octicons-arrow-right-24: Evaluate an existing method](contributing-a-method.md)

- :material-notebook-multiple: **Reproducible Notebooks**

    ---

    Three notebooks: a classical-planner walkthrough on released tasks, the
    paper results and the leaderboard. Re-run with `make docs-prepare`.

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

- You have a planner and need decoded-path scores under an identified
  qualified task and procedure contract.
- You want to compare planners on identical frozen tasks with paired geographic
  block intervals, explicit failure denominators and joint domain support. The
  historical protocol's mean-difference and multiple-testing utilities remain
  separate from this current comparison workflow.
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
