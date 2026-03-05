---
title: Generation Process
icon: material/chart-timeline-variant
---

# Generation Process

This section provides a **deep technical dive** into how PASSAGE generates pathfinding samples — from the mathematical foundations of sampling and cost models to the algorithmic details of obstacle placement and solver execution.

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Input
        T[JAXA Tiles<br>30 m/px GeoTIFF]
        C[Calibration<br>global min/max]
        K[Config<br>solvers, obstacles]
    end

    subgraph Sampling
        S1["1. Draw target elevation<br>uniformly in [e_min, e_max]"]
        S2[2. Select matching tile]
        S3[3. Extract random crop]
        S4[4. Place start/end markers]
    end

    subgraph Obstacles
        O1[5. Generate superellipse mask]
        O2[6. Validate coverage ≤ 30%]
    end

    subgraph Solving
        P1[7. Solve on free terrain]
        P2[8. Apply obstacle penalty]
        P3[9. Solve with obstacles]
    end

    subgraph Output
        OUT[10. Encode tensor + metadata<br>Write to disk]
    end

    T --> S1
    C --> S1
    K --> S1
    S1 --> S2 --> S3 --> S4
    S4 --> O1 --> O2
    O2 --> P1 --> P2 --> P3
    P3 --> OUT

    style S1 fill:#1565C0,color:#fff
    style O1 fill:#D32F2F,color:#fff
    style P1 fill:#2E7D32,color:#fff
    style OUT fill:#42A5F5,color:#fff
```

---

## Sections

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [:material-dice-multiple: Sampling Procedure](sampling.md)
The 6-step process for generating each sample — from elevation targeting to normalization, with full mathematical detail.
</div>

<div class="feature-card" markdown>
### [:material-scale-balance: Cost Models](cost-models.md)
Five distinct edge-weight functions that model different navigation priorities — elevation, distance, slope, energy, and absolute.
</div>

<div class="feature-card" markdown>
### [:material-shape: Obstacle Generation](obstacles.md)
Superellipse-based procedural obstacles with configurable shape distributions, rotation, and coverage control.
</div>

<div class="feature-card" markdown>
### [:material-graph: Solvers & Heuristics](solvers.md)
Three solver backends (NetworkX, pathfinding, Numba-JIT grid), A\* and Dijkstra algorithms, and three admissible heuristics.
</div>

<div class="feature-card" markdown>
### [:material-notebook: Interactive Notebooks](notebooks.md)
Explore the dataset visually through Jupyter notebooks — cost model comparisons, obstacle distributions, and solver analysis.
</div>

</div>
