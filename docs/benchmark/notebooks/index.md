---
title: "Notebooks"
icon: material/notebook-multiple
---

# Benchmark notebooks

Four notebooks cover the `passage-eval` side: an evaluator walkthrough, the paper results, the classical baselines and the results format. `make docs-prepare` executes them and exports standalone HTML files embedded below; a notebook that needs a local dataset copy is rendered unexecuted and flagged.

[Notebook export licenses](../../licenses/notebook-exports.md)

<div class="grid cards" markdown>

- :material-notebook-outline: **Evaluator walkthrough**

    ---

    Writes a planner in a few lines, scores it next to the exact A* planner on one released task, reads each of the six primary metrics, then scores both on a small validation sample. It needs the local dataset copy declared by `dataset_path`; without it, the page shows the notebook unexecuted.

    [:octicons-arrow-right-24: Open the notebook](./01-astar-walkthrough.md)

- :material-notebook-outline: **Paper results**

    ---

    Reads the per-task results of the two paper finalists released under `src/passage-eval/results/paper-results/`. They are not leaderboard entries.

    [:octicons-arrow-right-24: Open the notebook](./02-paper-results.md)

- :material-notebook-outline: **Classical baselines**

    ---

    Scores the three classical planners shipped with `passage-eval` (exact A*, greedy, random walk) on 64 validation tasks for each objective, with a distribution-free interval for each median regret. Reference points for reading the metrics, not benchmark results. Needs the local dataset copy.

    [:octicons-arrow-right-24: Open the notebook](./02-score-released-baselines.md)

- :material-notebook-outline: **Results format**

    ---

    Scores two classical planners, records them as `Submission` records in a temporary results folder and reads the table back: benchmark cell, six metrics, data revision and code version, without host or user names. Needs the local dataset copy.

    [:octicons-arrow-right-24: Open the notebook](./03-leaderboard.md)


</div>
