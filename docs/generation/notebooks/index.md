---
title: "Notebooks"
icon: material/notebook-multiple
---

# Dataset / solver notebooks

Top-level demo and analysis notebooks for the data-generation side. Each notebook is best-effort executed during `make docs-prepare`; notebooks that need a downloaded dataset fall back to their last stored outputs and are flagged with a warning admonition.

<div class="grid cards" markdown>

- :material-notebook-outline: **Demo (load + visualise)**

    ---

    Quick-start: load one PASSAGE sample at the requested resolution + split, decode the compressed tensor, render the elevation field, the markers, the obstacle mask, and the oracle paths. This is the canonical first notebook to run after `make download && make calibrate`.

    [:octicons-arrow-right-24: Open the notebook](./demo.md)

- :material-notebook-outline: **Obstacle generation**

    ---

    Walks through the PASSAGE obstacle generation pipeline: super-ellipse shape gallery, coverage histograms across configurations, and the pseudo-random seeding contract that guarantees per-sample reproducibility.

    [:octicons-arrow-right-24: Open the notebook](./obstacles.md)

- :material-notebook-outline: **Cost model comparison**

    ---

    Compare all five PASSAGE cost models — `distance`, `elevation`, `absolute_elevation`, `slope_uphill`, `energy` — on the same terrain crop, side-by-side. Includes the weight calibration explanation.

    [:octicons-arrow-right-24: Open the notebook](./costmodel.md)

- :material-notebook-outline: **Solver comparison**

    ---

    Side-by-side timing and quality comparison of three shortest-path solver back-ends used during data generation: NetworkX, the `pathfinding` package, and the PASSAGE grid back-end.

    [:octicons-arrow-right-24: Open the notebook](./solve.md)

- :material-notebook-outline: **A* inference timing**

    ---

    Reference deployment-side benchmark for the A* oracle implementation used as the dataset's labelling solver.

    [:octicons-arrow-right-24: Open the notebook](./astar-inference-timing.md)


</div>
