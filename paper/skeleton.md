# PASSAGE — Paper Skeleton

> **Target venue:** NeurIPS Datasets & Benchmarks Track (~9 pages + appendix)
> **Narrative voice:** single, unified

---

## Title

**PASSAGE: A Multi-Resolution Elevation-Based Pathfinding Dataset for Certifiable Machine Learning Systems**

*Alternative:* PASSAGE: Real-World Terrain Pathfinding Benchmarks from Satellite Elevation Data

---

## Abstract (~200 words)

| Element | Content |
|---------|---------|
| **Problem** | Learning-based pathfinding (surrogate models, RL, GNNs) lacks standardized benchmarks with continuous terrain costs, multiple resolutions, and certification-grade metadata. |
| **Dataset** | PASSAGE: 141,300 samples across 7 resolutions (64×64 → 4096×4096), derived from JAXA ALOS AW3D30 (30 m/px real-world elevation). |
| **Key features** | 5 physically-motivated cost models, 3 solver backends, superellipse procedural obstacles, stratified train/calibration/validation/test splits, full provenance metadata per sample. |
| **Design principles** | Certification-oriented (ARP6983/ED-324 alignment): bounded ODD, deterministic generation, calibration split for conformal prediction. |
| **Availability** | Open-source pipeline + Hugging Face Hub export with zstd-compressed Parquet shards. |
| **Baselines** | Solver timing comparison, obstacle distribution analysis, U-Net / GNN / PPO baselines. |

---

## 1. Introduction

### 1.1 Motivating Hook — HEMS (1 paragraph)

> Every year, Helicopter Emergency Medical Services (HEMS) across Europe perform over 300,000 rescue missions — racing through mountain passes, across valleys, and over ridgelines. For cardiac arrest or severe trauma, survival drops 7–10 % per minute of delay. The path the helicopter takes matters — yet pathfinding algorithms are still benchmarked on flat, synthetic grids with binary obstacles. **PASSAGE** was built to bridge this gap: real-world terrain, real costs, real complexity.

*Key references:* Michaels et al. 2019 (helicopter vs ground ambulance survival, 57 % mortality reduction); Snooks et al. 1996 (HEMS cost-effectiveness); Branas et al. 2005 (access to trauma centers).

### 1.2 Context: Learning-Based Pathfinding

- Supervised surrogate models (U-Net path prediction, cost-to-go estimation)
- Reinforcement learning for navigation (DQN, PPO, hierarchical RL)
- Graph neural networks for combinatorial optimization
- Differentiable planners: Neural A* (Yonetani et al. 2021), Value Iteration Networks (Tamar et al. 2016)
- **Common need:** large-scale, reproducible benchmarks with ground-truth paths

### 1.3 Gaps in Existing Benchmarks

| Gap | Existing state | PASSAGE |
|-----|---------------|---------|
| Resolution variety | Fixed (Dragon Age 512×512, StarCraft 1024×1024) | 7 levels: 64 → 4096 |
| Cost models | Binary obstacles only | 5 continuous terrain-based models |
| Sample counts | Hundreds to low thousands | 141,300 (configurable) |
| Certification metadata | None | Per-sample provenance, ODD definition, calibration split |
| Solver diversity | Single algorithm | 3 backends × 6 algorithms |
| Terrain realism | Video-game maps or random grids | Satellite elevation (JAXA ALOS AW3D30) |

### 1.4 Contributions (numbered list, 5 items)

1. **PASSAGE Dataset** — large-scale, multi-resolution pathfinding benchmark from real elevation data
2. **Procedural Obstacle Generation** — configurable superellipse (Lamé curve) obstacles with multiple shape distributions
3. **Multi-Backend Solver Framework** — 3 backends, 6 algorithms, 5 cost models
4. **Certification-Oriented Design** — bounded ODD, deterministic seeds, provenance metadata, calibration split
5. **Open-Source Toolkit** — complete pipeline from satellite tile download to Hugging Face Hub export

### 1.5 Paper Organization

Brief roadmap of sections 2–8.

---

## 2. Related Work

### 2.1 Grid-Based Pathfinding Benchmarks

- **Moving AI Lab** (Sturtevant 2012): Dragon Age, StarCraft, Warcraft III, Baldur's Gate maps; city/street discretizations; random/maze generators. _Limitations:_ binary passable/impassable, fixed resolution, no terrain cost gradients.
- **Random grid generators**: scalable but lack geographic realism.
- **Terrain maps** (Moving AI "terrain" set): discrete terrain types, but still categorical costs, not continuous elevation.
- _Contrast with PASSAGE:_ continuous elevation-derived costs, 7 resolutions, 141k samples.

### 2.2 Continuous Navigation & Robotics Datasets

- nuScenes (Caesar et al. 2020), Waymo Open Dataset (Sun et al. 2020): rich sensor data, continuous state space. Not suitable for discrete grid-based algorithm evaluation.
- Robot navigation (Mirowski et al. 2017): focus on visual navigation, not grid pathfinding.
- _Contrast:_ PASSAGE targets discrete grid algorithms while retaining real-world terrain.

### 2.3 Elevation / DEM-Based Data

- SRTM, ASTER GDEM, ALOS AW3D30: available for GIS but lack pathfinding ground truth, train/test splits, obstacle annotations.
- Existing GIS routing (GRASS GIS, QGIS): tool-oriented, not ML-benchmark-oriented.

### 2.4 Learning-Based Pathfinding

- **Supervised:** U-Net for path segmentation (Ronneberger et al. 2015 adapted); encoder-decoder cost-to-go estimation.
- **Value Iteration Networks** (Tamar et al. 2016, GVIN): differentiable planning modules.
- **Neural A*** (Yonetani et al. 2021): learned cost maps guiding classical A*. _Uses Moving AI Lab data._
- **GNNs for combinatorial optimization** (Cappart et al. 2021): message-passing on graph structures.
- **RL for navigation** (Mirowski et al. 2017; Schulman et al. 2017 PPO): sample efficiency and generalization challenges.
- **Learned heuristics** and algorithm selection: promising but dataset-dependent.

### 2.5 AI Certification Standards (brief)

- SAE ARP6983: process for AI-based aeronautical products.
- EUROCAE ED-324 (WG-114): ML guidance for aviation.
- Key requirements relevant to datasets: defined ODD, data quality, reproducibility, performance bounds.
- Conformal prediction (Vovk et al. 2005): coverage guarantees → needs calibration split.

---

## 3. Dataset Design

### 3.1 Data Source: JAXA ALOS AW3D30

- 30 m/px, near-global (±60° lat), GeoTIFF 3600×3600 tiles (1°×1°), meters above EGM96 geoid, ~5 m vertical RMSE.
- Selection rationale: global coverage, consistent quality, free for research.

### 3.2 Operational Design Domain (ODD)

Table defining bounds: geographic, elevation, grid resolutions, obstacle coverage, superellipse exponent range, connectivity.

### 3.3 Multi-Resolution Tiers

Table: 7 resolutions (64 → 4096), sample counts, grid cells, physical extent, complexity class.

- Design rationale: small resolutions for training, large for challenging evaluation.
- Skewed distribution enables computational scalability studies.

### 3.4 Dataset Splits

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 90 % | Model training |
| Calibration | 2 % | Conformal prediction, threshold tuning (certification) |
| Validation | 4 % | Hyperparameter selection |
| Test | 4 % | Final evaluation |

- **Geographic hold-out regions** (hold_out_mask.png): specific geographic tiles reserved exclusively for test split to prevent spatial leakage.

### 3.5 Data Representation

- Tensor format: `(H, W, 3)` float32 — channel 0: normalized elevation [0,1]; channel 1: markers (start=0, goal=1, background=−1); channel 2: obstacle mask (0/1).
- On-disk: zstd-compressed `.npy.zst` blobs.
- Metadata: YAML with full provenance (tile IDs, geo-coordinates, elevation stats, path data, timing breakdowns, solver config).

---

## 4. Generation Pipeline (detailed)

### 4.1 Pipeline Overview

```
Download → Calibrate → Generate → Export
```

Diagram: 4-stage pipeline with inputs/outputs for each stage.

### 4.2 Tile Download & Stitching

- URL construction for JAXA archives, CSV status tracking, resume support.
- Selective DSM file extraction from ZIP archives.
- Neighboring tile loading: for large resolutions (e.g., 4096×4096 = 122.9 km extent), multiple adjacent tiles must be stitched.
  - Handle ±180° longitude wrapping.
  - Handle latitude bounds (±60°).
  - Grid of neighbors centered on selected tile.

### 4.3 Calibration

- Parallel scan of all tiles → global min/max elevation.
- Per-tile statistics (min, max, mean, std) stored in `calibrate.json`.
- Used for consistent normalization across all samples.
- Incremental: supports resume if interrupted.

### 4.4 Sample Generation Algorithm (detailed)

**For each sample:**

1. **Elevation targeting:** Draw random target elevation from [global_min, global_max].
2. **Tile selection:** Filter tiles whose min/max spans the target elevation; sample one center tile. For test split: only tiles in hold-out mask. For other splits: exclude hold-out tiles.
3. **Neighbor loading:** Load grid of surrounding tiles (handle wrapping at ±180°). Find pixel closest to target elevation in center tile.
4. **Crop extraction:** Sample random R×R crop containing the target pixel.
5. **Marker placement:** Start at target pixel; goal at a distinct random pixel (ensure valid path exists on free terrain).
6. **Obstacle generation:** Run Algorithm 1 (superellipse placement) with marker-position protection.
7. **Pathfinding (free terrain):** Compute path on obstacle-free elevation map.
8. **Pathfinding (with obstacles):** Apply obstacle penalty (elevation + 10⁸) and recompute path.
9. **Persistence:** Save tensor (.npy.zst), metadata (.yaml), optional PNG visualization.

**Algorithm 1: Obstacle Generation via Superellipses**

```
Input: resolution R, config C
mask ← zeros(R, R)
target_ratio ← Uniform(0, C.ratio_max)
for attempt = 1 to C.max_attempts:
    candidate ← copy(mask)
    for i = 1 to C.candidates_per_mask:
        Sample n from C.distribution (log-uniform / uniform / beta / exponential)
        Sample a, b as fractions of R (within [min_size_ratio, max_size_ratio])
        Sample angle θ from config
        Sample center (cx, cy) uniformly in [0, R)²
        new_mask ← SuperellipseMask(R, (cy,cx), a, b, n, θ)
        if coverage(candidate ∨ new_mask) ≤ C.ratio_max:
            candidate ← candidate ∨ new_mask
    if coverage(candidate) ≥ target_ratio:
        return candidate
return mask  // fallback: no or partial obstacles
```

**Superellipse equation:**

$$\left(\frac{|x - c_x|}{a}\right)^n + \left(\frac{|y - c_y|}{b}\right)^n \leq 1$$

- n=1 → diamond, n=2 → ellipse, n>2 → squircle → rectangle
- Rotation by angle θ for orientation diversity
- Non-overlapping constraint enforced implicitly through coverage cap

**Shape distribution analysis:** table of log-uniform (default, favors round), uniform, beta (configurable skew), exponential (rapid decay).

### 4.5 Cost Models (5 models, detailed)

| Model | Edge weight formula | Physical interpretation |
|-------|-------------------|----------------------|
| **Elevation** | $w = d \cdot (1 + \alpha \cdot \bar{e}_{\text{dst}})$ | Penalizes high-altitude destinations |
| **Distance** | $w = d$ | Pure geometric distance (terrain-agnostic) |
| **Slope-uphill** | $w = d \cdot (1 + \alpha \cdot \max(0, \Delta e) / \Delta e_{\text{range}})$ | Penalizes climbing |
| **Energy** | $w = d + \alpha \cdot \max(0, \Delta e)$ | Simplified physical energy expenditure |
| **Absolute elevation** | $w = d \cdot (1 + \alpha \cdot |\bar{e}_{\text{dst}}|)$ | Penalizes elevation extremes (both high and low) |

Where:
- $d$: physical distance (30 m cardinal, 30√2 ≈ 42.43 m diagonal)
- $\alpha$: configurable weight (default 100.0)
- $\bar{e}$: normalized elevation ∈ [0, 1]
- $\Delta e$: elevation difference (dst − src)

**Obstacle enforcement:** masked cells get $e_{\text{masked}} = \max(e) + 10^8$, so all cost models naturally produce prohibitive edge weights without special-casing solvers.

### 4.6 Solver Backends & Algorithms

| Backend | Library | Algorithms | Best for |
|---------|---------|-----------|----------|
| **NetworkX** | `networkx` | A*, A* bidir., Dijkstra, Dijkstra bidir., Bellman-Ford, unweighted BFS | Flexibility, all cost models |
| **Pathfinding** | `pathfinding` | A*, Dijkstra, Best-First | Small grids, quick prototyping |
| **Grid** | NumPy + Numba JIT | grid_astar, grid_dijkstra | Large grids (≥1024), speed |

**Heuristics (for A*/Best-First):**
- Euclidean: $h = 30 \cdot \sqrt{(\Delta i)^2 + (\Delta j)^2}$ (recommended for 8-connectivity)
- Manhattan: $h = 30 \cdot (|\Delta i| + |\Delta j|)$ (for 4-connectivity)
- Chebyshev: $h = 30 \cdot \max(|\Delta i|, |\Delta j|)$

**Graph caching:** NetworkX backend reuses cached graph templates and updates edge weights per crop to reduce overhead.

### 4.7 Hugging Face Hub Export

- Zstandard (zstd) compression at configurable levels.
- Per-resolution chunk sizes → controlled shard granularity (see Appendix for chunk size estimation).
- Token-based auth via `.env.secret`.
- Retry/resume for robustness.
- Automatic dataset card generation.

---

## 5. Dataset Statistics & Experimental Validation

### 5.1 Global Statistics

- Total samples, per-resolution counts, split distributions.
- Elevation histogram (global, start-point, path-pixel).
- Tile coverage map (heatmap of which tiles are used).

### 5.2 Obstacle Coverage Distribution

- Histogram of actual obstacle ratios (should be near-uniform [0, 0.30] with slight right-skew from non-overlap constraints).
- Shape parameter (n) distributions across samples.

### 5.3 Solver Timing Benchmarks

- Log-log plot: median solve time vs. resolution for 3 backends.
- Expected: Grid ≈ 10× faster at 4096×4096; NetworkX dominates at small resolutions.
- Scalability factor: $t(2R) / t(R)$.

### 5.4 Cost Model Sensitivity

- Same terrain + markers, different cost models → different path morphologies.
- Visual comparison figure.
- Path divergence metrics (Hausdorff, Fréchet, DTW, Chamfer, area-between) between cost models.

### 5.5 Path Divergence Metrics

- Normalized by marker distance for resolution-independent comparison.
- Clean path vs. obstacle path: quantify obstacle impact.
- Path vs. straight line: quantify terrain influence.

### 5.6 Baseline Learning Models

| Model | Architecture | Task | Key metric |
|-------|-------------|------|------------|
| **U-Net** | Encoder-decoder CNN | Path segmentation (binary mask) | Pixel Acc, IoU |
| **GCN** | 4-layer Graph Convolutional Network | Node-level path prediction | Pixel Acc, IoU |
| **PPO** | Proximal Policy Optimization | Navigation on 64×64/128×128 | Success rate |

Table: Baseline results on 256×256 test set.

---

## 6. Research Directions Enabled by PASSAGE

### 6.1 Generalization & Distribution Shift

- **Resolution transfer:** Train on 64×64, evaluate on 256×256 or 1024×1024. Can models generalize across grid scales?
- **Elevation extrapolation:** Train on mid-elevation terrain, test on mountain peaks or deep valleys.
- **Obstacle density shift:** Train on low coverage (0–10 %), test on high coverage (20–30 %).
- **Geographic hold-out:** Evaluate on held-out geographic regions unseen during training.

### 6.2 Certification-Ready ML Models

- **Conformal prediction:** Use calibration split to construct prediction intervals with provable coverage guarantees. Open question: coverage stability under resolution shift.
- **Uncertainty quantification:** Bayesian neural networks, deep ensembles, evidential deep learning. PASSAGE provides the sample counts and split structure needed for calibration experiments.
- **Formal verification:** Lipschitz bounds for neural path predictors; certified robustness to elevation perturbations.
- **OOD detection:** Detect when terrain or obstacle patterns fall outside the training ODD.

### 6.3 Scalable Architectures

- **Efficient transformers:** Linear attention, sparse patterns for high-resolution (2048×2048+) inputs.
- **Neural algorithmic reasoning:** Differentiable dynamic programming, pointer networks for variable-length path output.
- **Hierarchical planning:** Multi-scale approaches — coarse path on 64×64, refined on 512×512 or 1024×1024.
- **Diffusion models for path generation:** Path as a structured output; can denoising produce near-optimal trajectories?

### 6.4 Hybrid Neural-Classical Systems

- **Learned heuristics for A*:** Train a neural network to predict a better heuristic; benchmark against PASSAGE ground truth.
- **Neural cost map refinement:** Learn residual corrections to physical cost models.
- **Algorithm selection:** Predict which solver backend/algorithm is fastest for a given instance (meta-learning).

### 6.5 Multi-Agent & Temporal Extensions

- **Multi-agent pathfinding (MAPF):** Collision-free paths for multiple agents on the same elevation map.
- **Dynamic obstacles:** Time-varying obstacle masks; re-planning evaluation.
- **Ordered waypoints:** Intermediate points that must be visited in sequence (TSP-like extension).
- **Flight model constraints:** Turn radius, climb/descent rate limits, energy budgets — making paths physically flyable for HEMS.

### 6.6 Fairness, Geography & Terrain Diversity

- **Geographic stratification:** Analyze model performance by terrain type (mountain, desert, coastal, urban fringe).
- **Bias audit:** Do models perform worse on under-represented terrain types?
- **Terrain-aware data augmentation:** Synthetic elevation perturbations to improve diversity.

---

## 7. Limitations

- **Static environments:** No dynamic obstacles or temporal changes.
- **Single-agent focus:** No multi-agent collision avoidance.
- **Simplified cost models:** No wind, vehicle dynamics, fuel curves.
- **Resolution–sample trade-off:** Only 100 samples at 4096×4096 (computational cost).
- **Geographic bias:** Random sampling may not uniformly cover all terrain types.
- **Obstacle realism:** Superellipses are geometric; do not model real-world objects (buildings, restricted airspace).
- **Elevation-only terrain:** No vegetation, land-use, or atmospheric data layers.

---

## 8. Broader Impact & Ethics

- **Positive:** Search & rescue optimization, disaster response, environmental monitoring, accessible routing.
- **Dual-use risks:** Military pathfinding applications; responsible AI principles encouraged.
- **Privacy:** 30 m resolution insufficient to identify individuals; derived applications should assess higher-resolution risks.
- **Environmental:** Optimized routing may increase access to fragile areas.
- **Carbon footprint:** Pre-generated datasets on HF Hub reduce redundant computation.
- **Data licensing:** ALOS AW3D30 terms permit research use; users should verify for commercial use.

---

## 9. Conclusion

- Summarize 5 contributions.
- Highlight what PASSAGE enables that was not previously possible.
- Future work: waypoints, multi-agent, flight constraints, richer terrain layers.
- Availability: dataset on HuggingFace, code on GitHub (ThalesGroup/passage).

---

## Appendix

### A. Full Configuration Schema

Excerpt of `config/config.yaml` with all parameters.

### B. CLI Reference

Commands: `passage download`, `passage calibrate`, `passage generate`, `passage export`.

### C. Extended Experimental Results

- Baseline results across all 7 resolutions.
- Ablation: cost model impact on path morphology.
- Ablation: obstacle density vs. path deviation.

### D. Chunk Size Estimation

Formula and script for computing optimal HF shard sizes.

### E. Reproducibility Checklist

- [ ] Code released (MIT license)
- [ ] Dataset on Hugging Face Hub
- [ ] Config files included
- [ ] Seeds specified
- [ ] Hardware requirements documented
- [ ] Per-sample provenance metadata

### F. Datasheet for PASSAGE

Following Gebru et al. 2021 "Datasheets for Datasets" template:
- Motivation, Composition, Collection process, Preprocessing, Uses, Distribution, Maintenance.

---

## Key References (indicative, ~30–40 total)

| Category | Key references |
|----------|---------------|
| **Data source** | Tadono et al. 2014 (ALOS AW3D30) |
| **Pathfinding benchmarks** | Sturtevant 2012 (Moving AI Lab) |
| **Classical algorithms** | Hart et al. 1968 (A*); Dijkstra 1959 |
| **Supervised pathfinding** | Ronneberger et al. 2015 (U-Net); Yonetani et al. 2021 (Neural A*) |
| **VINs / differentiable planning** | Tamar et al. 2016 (VIN) |
| **GNNs** | Kipf & Welling 2017 (GCN); Cappart et al. 2021 (combinatorial optimization) |
| **RL** | Schulman et al. 2017 (PPO); Mirowski et al. 2017 |
| **MAPF** | Stern et al. 2019 |
| **Conformal prediction** | Vovk et al. 2005 |
| **Certification** | SAE ARP6983; EUROCAE ED-324 |
| **Autonomous driving datasets** | Caesar et al. 2020 (nuScenes); Sun et al. 2020 (Waymo) |
| **HEMS / air ambulance** | Michaels et al. 2019; Snooks et al. 1996; Branas et al. 2005 |
| **Datasheets** | Gebru et al. 2021 (Datasheets for Datasets) |
| **Superellipses** | Lamé curves (mathematical background) |

---

## Figures Plan

| # | Description | Section |
|---|-------------|---------|
| 1 | Pipeline overview diagram (Download → Calibrate → Generate → Export) | §4.1 |
| 2 | Sample gallery: 7 resolutions, elevation + obstacles + path | §3 or §5 |
| 3 | Superellipse shape gallery (n=1, 2, 4, 10, 80) | §4.4 |
| 4 | Obstacle generation process (iterative placement) | §4.4 |
| 5 | Cost model comparison: same terrain, 5 different paths | §5.4 |
| 6 | Solver timing: log-log plot by resolution | §5.3 |
| 7 | Obstacle coverage histogram | §5.2 |
| 8 | Elevation & path-length distributions | §5.1 |
| 9 | Geographic tile coverage heatmap | §5.1 |
| 10 | Baseline results (bar chart) | §5.6 |

---

## Tables Plan

| # | Description | Section |
|---|-------------|---------|
| 1 | ODD specification | §3.2 |
| 2 | Resolution tiers (samples, grid cells, extent) | §3.3 |
| 3 | Split ratios | §3.4 |
| 4 | Cost model formulas | §4.5 |
| 5 | Solver backends & algorithms | §4.6 |
| 6 | Shape distributions for n | §4.4 |
| 7 | Baseline results (Pixel Acc, IoU, Success%) | §5.6 |
| 8 | Solver timing comparison | §5.3 |

---

## Writing Notes

- **Page budget (NeurIPS D&B):** ~9 pages main, unlimited appendix & references.
- **Emphasis:** Dataset design rationale + generation pipeline internals + breadth of enabled research, rather than heavy ML experiments.
- **Tone:** Accessible to ML audience; minimize GIS/aerospace jargon; define terrain-specific terms on first use.
- **Differentiation claim:** "First multi-resolution, elevation-based pathfinding dataset with certification-grade metadata and procedural obstacle generation."
