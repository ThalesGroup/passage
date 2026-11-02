---
title: Generation Process
icon: material/chart-timeline-variant
---

# Generation Process

This section provides a **deep technical dive** into how PASSAGE generates its scenes, from the sampling of each scene and the cost models to obstacle placement and the exact reference solver. It describes the generator that built **PASSAGE v1.0**: what `passage generate` runs with the shipped `config/config.yaml`.

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Input
        T["JAXA AW3D30 tiles<br>1°×1°, 3600×3600 cells of 30 m"]
        C["Calibration<br>global min/max: −430.0 / 8767.0 m"]
        K["Config<br>splits, objectives, obstacles"]
    end

    subgraph Sampling
        S1["1. Draw a center tile<br>among the split's candidates"]
        S2["2. Draw a target elevation;<br>nearest cell = start"]
        S3["3. Place the R×R crop around the start<br>(split, guard ring, nodata, range checks)"]
        S4["4. Draw the goal<br>among the other cells"]
    end

    subgraph Obstacles
        O1["5. Superellipse mask<br>coverage ≤ target ≤ 30 %, markers kept free"]
    end

    subgraph Solving
        P1["6. Write the float32 tensor<br>and metadata"]
        P2["7. Exact float64 A* reference:<br>3 objectives × free / obstacles blocked"]
    end

    subgraph Output
        OUT["8. Write routes and the scene receipt"]
    end

    T --> S1
    C --> S2
    K --> S1
    S1 --> S2 --> S3 --> S4
    S4 --> O1
    O1 --> P1 --> P2
    P2 --> OUT

    style S1 fill:#1565C0,color:#fff
    style O1 fill:#D32F2F,color:#fff
    style P2 fill:#2E7D32,color:#fff
    style OUT fill:#42A5F5,color:#fff
```

Each scene yields six route variants, each an exact optimal route, never an approximation. A crop whose obstacles disconnect the start from the goal is rejected before any label is computed, and the scene draws its next candidate crop (see [Blocked Cells and Connected Endpoints](obstacles.md#blocked-cells-and-connected-endpoints)).

---

## Sections

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [:material-dice-multiple: Sampling Procedure](sampling.md)
How each scene is drawn: center tile, start cell, crop, goal and normalization, plus the geographic split rules (test-only South America hold-out, 5°×5° blocks, one-tile guard ring).
</div>

<div class="feature-card" markdown>
### [:material-scale-balance: Cost Models](cost-models.md)
Five crop-relative step-cost functions. PASSAGE v1.0 uses three of them: `elevation`, `slope_uphill` and `energy`.
</div>

<div class="feature-card" markdown>
### [:material-shape: Obstacle Generation](obstacles.md)
Superellipse obstacles with random shape, size and rotation, covering up to 30 % of the crop. Obstacle cells are blocked.
</div>

<div class="feature-card" markdown>
### [:material-graph: Solvers & Heuristics](solvers.md)
The exact float64 A\* reference solver that labels v1.0, and three alternative backends (NetworkX, pathfinding, Numba grid) not used by the release.
</div>

<div class="feature-card" markdown>
### [:material-notebook: Interactive Notebooks](notebooks/index.md)
Explore released scenes in Jupyter notebooks: the three objectives, obstacle masks and detours, and exact re-solves of the stored labels.
</div>

</div>
