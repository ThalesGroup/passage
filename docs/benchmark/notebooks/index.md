---
title: "Notebooks"
icon: material/notebook-multiple
---

# Benchmark notebooks

Four notebooks cover the `passage-eval` side: an evaluator walkthrough, historical validation results, classical baselines and the results format. Open each notebook source to view or run it locally.

[Notebook rendering and licenses](../../licenses/notebook-exports.md)

<div class="grid cards" markdown>

- :material-notebook-outline: **Evaluator walkthrough**

    ---

    Writes a planner in a few lines, scores it next to the exact A* planner on one released task, reads each of the six primary metrics, then scores both on a small validation sample. It reads the local dataset copy named by `dataset_path`, or the Hub at the pinned `dataset_revision` when `dataset_path = None`.

    [:octicons-arrow-right-24: Open the notebook](./01-astar-walkthrough.md)

- :material-notebook-outline: **Historical validation results**

    ---

    Tabulates the per-seed aggregate metrics (`metrics.json`) of earlier 256×256 validation runs with older Seg and Reg baseline procedures, stored under `src/passage-eval/results/paper-results/`, and their seed means. The source notebook uses its original labels. It does not reconstruct the [camera-ready paper results](../candidates/index.md) or populate the leaderboard.

    [:octicons-arrow-right-24: Open the notebook](./02-paper-results.md)

- :material-notebook-outline: **Classical baselines**

    ---

    Scores the three classical planners shipped with `passage-eval` (exact A*, greedy, random walk) on 64 validation tasks for each objective, with a distribution-free interval for each median regret. Reference points for reading the metrics, not benchmark results. Reads the local dataset copy, or the Hub at the pinned revision.

    [:octicons-arrow-right-24: Open the notebook](./02-score-released-baselines.md)

- :material-notebook-outline: **Results format**

    ---

    Scores two classical planners, records them as `Submission` records in a temporary results folder and reads the table back: benchmark cell, six metrics, data revision and code version, without host or user names. Reads the local dataset copy, or the Hub at the pinned revision.

    [:octicons-arrow-right-24: Open the notebook](./03-leaderboard.md)


</div>
