---
title: "Notebooks"
icon: material/notebook-multiple
---

# Dataset / solver notebooks

Top-level demo and analysis notebooks for the data-generation side. Each notebook is best-effort executed during `make docs-prepare`; notebooks that need a downloaded dataset fall back to their last stored outputs and are flagged with a warning admonition.

<div class="grid cards" markdown>

- :material-notebook-outline: **Getting started**

    ---

    The shortest path from the dataset to a training example: load one released scene, decode its tensor, draw a reference route and turn it into a segmentation target.

    [:octicons-arrow-right-24: Open the notebook](./demo.md)

- :material-notebook-outline: **Obstacles**

    ---

    Obstacle masks from least to most covered, drawn coverage against its target, and the detour that obstacles force on the optimal route over a whole validation split.

    [:octicons-arrow-right-24: Open the notebook](./obstacles.md)

- :material-notebook-outline: **The three objectives**

    ---

    What the `elevation`, `slope_uphill` and `energy` objectives reward on one scene, their elevation profiles, and the extra cost of a route optimised for the wrong objective.

    [:octicons-arrow-right-24: Open the notebook](./costmodel.md)

- :material-notebook-outline: **Exact re-solve**

    ---

    Solves stored labels again with the exact float64 A* reference, compares costs and routes, shows why the search budget matters, and repeats the check on a sample of validation scenes.

    [:octicons-arrow-right-24: Open the notebook](./solve.md)

- :material-notebook-outline: **Timing a planner**

    ---

    Times the canonical A* planner of `passage-eval` with the practices that make a timing meaningful: warm-up, repetitions, percentiles and a stated machine.

    [:octicons-arrow-right-24: Open the notebook](./astar-inference-timing.md)


</div>
