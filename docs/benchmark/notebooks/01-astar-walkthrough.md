---
title: "A* walkthrough — `Planner` ABC end-to-end"
hide:
  - toc
---

# A* walkthrough — `Planner` ABC end-to-end

Implements the canonical cross-family abstraction end-to-end: score the bundled `AStarPlanner` through the `Planner` ABC, emit one run bundle to `outputs/passage-eval/results/`. This is the first notebook to run on a fresh checkout — it doubles as an integration test for the on-disk leaderboard layout.

!!! info "Source notebook"

    `notebooks/passage-eval/01_astar_walkthrough.ipynb` — re-execute with `make docs-prepare` after editing.

<iframe
    src="../../../gen/notebooks/01_astar_walkthrough.html"
    width="100%"
    height="1100"
    style="border:1px solid #eee; border-radius:6px; background:white;"
    loading="lazy">
</iframe>
