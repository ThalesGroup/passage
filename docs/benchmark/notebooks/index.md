---
title: "Notebooks"
icon: material/notebook-multiple
---

# Benchmark notebooks

Three reproducible notebooks walk through the canonical `passage-eval` workflow end-to-end. Each notebook is executed during `make docs-prepare` and exported as a standalone HTML file embedded below.

<div class="grid cards" markdown>

- :material-notebook-outline: **A* walkthrough**

    ---

    Implements the canonical cross-family abstraction end-to-end: score the bundled `AStarPlanner` through the `Planner` ABC, emit one run bundle to `outputs/passage-eval/results/`. This is the first notebook to run on a fresh checkout — it doubles as an integration test for the on-disk leaderboard layout.

    [:octicons-arrow-right-24: Open the notebook](./01-astar-walkthrough.md)

- :material-notebook-outline: **Score released baselines**

    ---

    Replay the released vision (`seg`, `reg`) and graph (Edge-MLP) per-sample NPZ artefacts through `passage_eval.io.predictions`. Each (config × seed × host_accelerator) cell becomes one leaderboard row in `outputs/passage-eval/results/runs/r256/...`.

    [:octicons-arrow-right-24: Open the notebook](./02-score-released-baselines.md)

- :material-notebook-outline: **Leaderboard visualisation**

    ---

    Aggregate everything emitted by the previous two notebooks and produce the four canonical figures: bar chart of CR-median per (method × accelerator), method × accelerator latency heatmap, accuracy-vs-latency Pareto frontier, pairwise verdict heatmap.

    [:octicons-arrow-right-24: Open the notebook](./03-leaderboard.md)


</div>
