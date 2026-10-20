---
title: "Notebooks"
icon: material/notebook-multiple
---

# Benchmark notebooks

Three notebooks cover the `passage-eval` side: a classical-planner walkthrough on released tasks, the paper results and the leaderboard. `make docs-prepare` executes them and exports standalone HTML files embedded below; a notebook that needs a local dataset copy is rendered unexecuted and flagged.

<div class="grid cards" markdown>

- :material-notebook-outline: **A* walkthrough**

    ---

    Reads one released task from a local copy of PASSAGE v1.0, renders its stored routes and runs a bounded exact-reference diagnostic with the classical planner. It needs the local dataset copy declared by `dataset_path`; without it, the page shows the notebook unexecuted.

    [:octicons-arrow-right-24: Open the notebook](./01-astar-walkthrough.md)

- :material-notebook-outline: **Paper results**

    ---

    Reads the per-task results of the two paper finalists released under `src/passage-eval/results/paper-results/`. These are the historical results of the original study, reported as obtained: they were not recomputed on the published dataset and are not leaderboard entries.

    [:octicons-arrow-right-24: Open the notebook](./02-paper-results.md)

- :material-notebook-outline: **Leaderboard**

    ---

    Reads the released leaderboard. It has no entries yet: the first entries will be the two paper finalists retrained on PASSAGE v1.0 and scored with `passage-eval`.

    [:octicons-arrow-right-24: Open the notebook](./03-leaderboard.md)


</div>
