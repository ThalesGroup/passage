---
title: "A* walkthrough — classical planner on released tasks"
hide:
  - toc
---

# A* walkthrough — classical planner on released tasks

Reads one released task from a local copy of PASSAGE v1.0, renders its stored routes and runs a bounded exact-reference diagnostic with the classical planner. It needs the local dataset copy declared by `dataset_path`; without it, the page shows the notebook unexecuted.

!!! info "Source notebook"

    `notebooks/passage-eval/01_astar_walkthrough.ipynb` — re-execute with `make docs-prepare` after editing.

!!! warning "This notebook was rendered without executing"

    `make docs-prepare` could not execute the source notebook on this
    checkout (execution skipped: the walkthrough needs a local dataset copy). The HTML below shows the most recent stored
    outputs, which may be stale or empty. Re-run the notebook locally
    after fetching the data it needs (e.g.\ `make download && make
    calibrate`) to refresh.

<iframe
    src="../../../gen/notebooks/01_astar_walkthrough.html"
    width="100%"
    height="1100"
    style="border:1px solid #eee; border-radius:6px; background:white;"
    loading="lazy">
</iframe>
