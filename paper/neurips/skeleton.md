# PASSAGE NeurIPS 2026 Evaluations & Datasets Track Skeleton

Last updated: 2026-03-17

## Submission context

- Target venue: NeurIPS 2026 Evaluations & Datasets Track
- Paper mode: anonymous submission
- Main-paper budget: 9 content pages
- Extra pages allowed: references, optional appendices, and mandatory checklist
- Important note: as of 2026-03-17, `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets` is still a placeholder page, so this skeleton is aligned primarily to the live 2026 Main Track Handbook plus the 2026 E&D FAQ / data-hosting guidance and the 2025 E&D call/blog for track-shape expectations
- Preferred contribution type: a use-inspired dataset-and-benchmark paper for safety-relevant evaluation, not a methods paper

## Compliance snapshot to honor while drafting

- Use the official NeurIPS 2026 LaTeX style and keep the paper anonymous, including linked artifacts and code.
- Do not include acknowledgments in the submission version.
- Keep the main paper to 9 content pages; put references, technical appendix material, and checklist after the main text.
- Include the NeurIPS checklist in the compiled PDF.
- If external artifacts are linked for review, they must support anonymous browsing.
- For a datasets/benchmarks paper, reviewers will expect a clear release path, reproducibility instructions, and strong documentation of assets, licenses, and limitations.
- The D&B FAQ and hosting guidance encourage structured metadata such as Croissant and stable archival hosting for accepted artifacts.
- The 2025 E&D call/blog strongly suggest that the paper should read like a benchmark paper with lifecycle thinking, not just an asset dump.
- Trustworthiness language must stay bounded: PASSAGE can support an assurance-oriented workflow and evidence package for learned airborne advisory modules, but the paper should not claim that the dataset itself "achieves certification."
- As of 2026-03-17, the most defensible wording is that PASSAGE is being shaped to align with current EASA and FAA AI-assurance thinking and with the future `ARP6983` learning-assurance standard expected in 2026, not that it already implements a finalized published standard.

## One-sentence positioning

PASSAGE should be framed as a reproducible, trustworthiness-oriented benchmark and transparent generation pipeline for terrain-aware learned advisory and surrogate tasks on real elevation data for bounded airborne use cases, not as a new autonomous flight-control algorithm.

## Recommended title options

1. PASSAGE: A Trustworthiness-Oriented Real-Terrain Benchmark for Learned Airborne Routing and Surrogate Modeling
2. PASSAGE: Benchmarking Terrain-Aware Airborne AI with Real Elevation Data, Transparent Generation, and Reproducible Provenance
3. PASSAGE: A Multi-Resolution Real-Terrain Benchmark for Airborne Advisory AI, Graph Learning, and Surrogate Models

## Draft abstract

Benchmarking learned decision-making for airborne systems remains difficult because widely used planning and surrogate-modeling testbeds are dominated by synthetic or game-derived grids that omit real terrain structure, explicit geographic hold-outs, raw-data provenance, and assurance-oriented release artifacts. We present PASSAGE, an open-source dataset and benchmark generation pipeline built from real elevation data derived from JAXA ALOS AW3D30 and designed for bounded airborne AI use cases such as terrain-aware route advisory, contingency rerouting support, and learned surrogates that accelerate deterministic planners. PASSAGE currently defines a default 14,130-sample benchmark configuration spanning seven resolutions from 64x64 to 4096x4096 cells, five terrain cost models, procedural superellipse obstacles, deterministic train/calibration/validation/test splits with a geographic hold-out policy, solver-derived reference trajectories, compressed tensor exports, and structured metadata suitable for reproducible analysis. The benchmark is intentionally multi-paradigm: it allows to support supervised learning with CNN- or transformer-style image formulations, graph-based learning with GNNs or reinforcement-learning studies under a surrogate modeling pattern of a legacy algorithm considered as the oracle. A core design choice is full transparency of both the released benchmark artifacts and the generator used to create them, including data lineage, preprocessing logic, and split definitions. PASSAGE is shared with the aim of helping build trustworthiness evidence toward the levels of assurance expected by EASA and the FAA, including future `ARP6983`/`ED-324`-style expectations, while supporting pioneers who want to embed AI onboard aircraft and, more broadly, in safety-critical applications.

## Paper strategy

The strongest version of this paper is dataset-first and benchmark-first:

- Lead with the evaluation gap: synthetic grids and game maps do not capture real terrain costs, geographic generalization, or provenance.
- Present PASSAGE as infrastructure for evaluation, surrogate modeling, and trustworthiness-oriented benchmarking.
- Use the airborne-assurance story as a bounded systems motivation, centered on advisory or surrogate components rather than autonomous flight control.
- Avoid strong “save lives” language unless domain validation or operational evidence is added.
- Spend most of the page budget on benchmark construction, evaluation protocol, and evidence that the benchmark is useful.

## Trustworthiness-oriented narrative

Primary system story:

- PASSAGE should target bounded airborne AI constituents such as terrain-aware route advisory, contingency rerouting support, route-cost prediction, or planner-acceleration surrogates.
- The learned component should be framed as advisory or acceleration logic inside a larger system that still contains deterministic planners, monitors, constraints, and human or certified-system oversight.
- The paper should repeatedly distinguish between certifying an aircraft function and sharing a benchmark that helps build trustworthiness evidence for a learned component inside an approval-oriented workflow.

Most defensible positioning:

- `PASSAGE is shared to help build trustworthiness evidence for bounded learned airborne components.`
- `PASSAGE helps operationalize emerging learning-assurance expectations by making data, generators, splits, thresholds, and lineage auditable.`
- `PASSAGE can support future ARP6983-aligned evidence workflows once the final standard is published and mapped in detail.`

Wording to avoid:

- `PASSAGE achieves EASA certification.`
- `PASSAGE is fully ARP6983-compliant today.`
- `PASSAGE validates autonomous aircraft control.`

## Supported AI problem formulations

### Supervised tensor-as-image learning

- Treat terrain tensors as images with start, goal, and obstacle channels.
- Benchmark CNN, U-Net, and vision-transformer models for cost-map prediction, waypoint heatmaps, path masks, or next-step policies.
- Evaluate geographic generalization, cross-resolution transfer, calibration, and robustness to obstacle injection.

### Graph learning

- Convert grids to graphs for GNN-based cost-to-go prediction, learned heuristics, path ranking, or graph-level route scoring.
- Compare graph encodings with image encodings under identical train/test splits.

### Reinforcement learning

- Use PASSAGE as an offline or simulator-backed benchmark for terrain-aware navigation policies.
- Restrict claims to benchmarked decision quality, safety constraints, and generalization under bounded reward definitions rather than operational autonomy.

### Complex surrogate modeling

- Learn surrogates for route cost, feasibility, solver runtime, or near-optimal path structure.
- Position this as a realistic hard surrogate task because the input-output mapping is multimodal, geography-dependent, and sensitive to cost-model choices.
- Emphasize that surrogate quality matters for trustworthiness-oriented deployment only when paired with monitors, thresholds, and fallback logic.

### Trustworthiness-oriented system architecture

- Recommend a reference architecture of `learned model + deterministic verifier/planner + threshold monitor + human or certified-system oversight`.
- Use PASSAGE to evaluate not only raw prediction accuracy, but also bounded-error behavior, abstention, fallback triggering, and monitor coverage.

## Claims that are safe today

- PASSAGE is built from real elevation data and provides a reproducible generation pipeline.
- The codebase supports seven resolutions, five cost models, multiple solver backends, procedural obstacles, and deterministic split assignment.
- The export pipeline produces Hugging Face-compatible Parquet shards with structured metadata.
- The benchmark can support studies of scaling, solver behavior, geographic generalization, and complex surrogate modeling.
- PASSAGE can be framed as support for trustworthiness-building workflow studies for bounded learned airborne components.
- PASSAGE can support supervised tensor-as-image learning, graph learning, reinforcement learning, and surrogate-modeling research if suitable baselines are added.

## Claims that should not appear until supported by experiments or cleanup

- Any claim of state-of-the-art learned planner performance.
- Any claim that the benchmark already demonstrates strong ML surrogate baselines unless those baselines are actually implemented and reported.
- Any claim that the default benchmark size is `141,300`; the current config sums to `14,130`.
- Any strong deployment or healthcare outcome claim without stakeholder validation.
- Any claim that PASSAGE alone is sufficient for EASA or FAA certification.
- Any claim of full `ARP6983` implementation or compliance before the final standard is published and the paper includes a detailed requirement-to-artifact mapping.

## Recommended main-paper structure

### 1. Introduction

Goal:
- establish the benchmark gap and state clearly that the contribution is a benchmark, not a new planner

Content:
- why current pathfinding evaluation is too synthetic
- why real terrain, multiple scales, and explicit split design matter
- what PASSAGE contributes
- concise contributions list

Suggested contribution bullets:
- a real-terrain, multi-resolution benchmark from ALOS AW3D30 for bounded airborne AI evaluation
- standardized task construction with explicit train / calibration / validation / test splits and geographic hold-out
- multiple terrain cost models and obstacle variants in one common benchmark
- auditable generation, metadata, raw-data lineage, and export pipeline
- a benchmark protocol spanning classical solvers, supervised learning, graph learning, reinforcement learning, and surrogate modeling
- a trustworthiness-oriented use case showing how PASSAGE can support an assurance workflow for learned airborne advisory or surrogate components

### 2. Related Work and Track Positioning

Goal:
- show awareness of both pathfinding literature and NeurIPS D&B expectations

Subsections:
- classical grid pathfinding benchmarks: Moving AI, game-map benchmarks
- learned planning on grids: VIN, GVIN, Neural A*, learned heuristics
- terrain / GIS routing data sources versus ML-ready benchmarks
- aviation AI assurance and trustworthiness guidance: EASA AI Concept Paper Issue 2, EASA AI Roadmap 2.0, FAA AI Safety Assurance Roadmap, and EASA NPA 2025-07
- benchmark-design papers from NeurIPS D&B for inspiration on structure and release expectations

Specific papers to consider citing:
- Sturtevant 2012 Moving AI benchmarks
- Tamar et al. 2016 Value Iteration Networks
- Yonetani et al. 2021 Neural A*
- GameTraversalBenchmark (NeurIPS 2025) for benchmark-paper structure
- FairJob (NeurIPS 2024 D&B) for real-world dataset packaging and Croissant metadata example
- Croissant: A Metadata Format for ML-Ready Datasets (NeurIPS 2024 D&B)
- EASA Artificial Intelligence Concept Paper Issue 2
- FAA Roadmap for Artificial Intelligence Safety Assurance Version I

### 3. Benchmark Design

Goal:
- define exactly what a PASSAGE task is

Must include:
- source data: JAXA ALOS AW3D30
- release mode for raw versus derived data, plus source-data manifests if direct redistribution is constrained
- task formulation: shortest-path search on elevation grids with start/goal markers
- resolutions and physical extents
- default sample counts and split ratios
- cost models
- obstacle generation
- geographic hold-out design
- intended airborne use cases, operational design domain assumptions, non-goals, and why transparent data-plus-generator release matters for trust building

Important table:
- comparison against prior benchmarks
  - real terrain
  - multi-resolution
  - continuous cost models
  - explicit geographic split design
  - structured metadata
  - release artifacts

### 4. Generation Pipeline and Data Representation

Goal:
- make the benchmark auditable and reproducible

Subsections:
- download and tile acquisition
- source-data manifests, retrieval scripts, and provenance logging
- calibration and normalization
- crop sampling and target-elevation anchoring
- marker placement
- obstacle generation via superellipses
- solver execution and path storage
- export format and metadata schema
- assurance-oriented artifact preservation: seeds, split manifest, version manifest, threshold tables, benchmark hashes, and generator provenance

Must show:
- enough detail that reviewers can understand how samples are generated
- why leakage is controlled
- where determinism and seeds enter

### 5. Evaluation Protocol

Goal:
- make the benchmark scientifically useful, not just available

Subsections:
- benchmark tasks
  - solver scaling across resolutions
  - cost-model sensitivity
  - obstacle sensitivity
  - geographic generalization
  - cross-resolution transfer for learned methods
  - trustworthiness-oriented surrogate use case for bounded airborne advisory support
- metrics
  - path cost
  - path length
  - runtime
  - success rate
  - obstacle-crossing rate
  - path divergence between free and obstacle-aware solutions
  - calibration / reliability metrics
  - bounded-error threshold metrics for acceptance and fallback
- baselines
  - classical: A*, Dijkstra, maybe bidirectional variants
  - learned image-style: at least one CNN / U-Net baseline and one transformer-style baseline if feasible
  - learned graph-style: at least one GNN or learned-heuristic baseline
  - RL: at least one offline or simulator-backed policy benchmark if feasible
  - surrogate: at least one route-cost or planner-acceleration model paired with a deterministic verifier
- protocol details
  - seeds
  - hardware
  - train/validation/calibration/test use
  - statistical reporting
  - threshold-setting and update-policy details for trustworthiness-oriented evaluation

### 6. Benchmark Analyses and Results

Goal:
- prove the benchmark is interesting and nontrivial

Minimum result set to include:
- solver runtime scaling with resolution
- distributional properties of elevation, slopes, obstacle coverage, and path lengths
- evidence that geographic hold-out is materially harder than random split evaluation
- qualitative examples where cost models induce very different paths
- benchmark diversity diagnostics
- one bounded airborne use case showing a learned surrogate or advisory model plus deterministic fallback or verifier logic and transparent evidence packaging

If time permits:
- baseline learned model performance
- cross-resolution transfer results
- ablations on obstacle distributions and hold-out policy
- calibration curves and threshold analyses for trustworthiness-oriented acceptance criteria

### 7. Release, Reproducibility, and Maintenance

Goal:
- match what D&B reviewers like to see

Include:
- code and data release plan
- raw-data release plan or, if needed, source-data manifests plus download-and-rebuild scripts
- artifact hosting plan for anonymous review and post-acceptance release
- versioning and benchmark maintenance plan
- metadata plan, ideally including Croissant
- license and terms for source and derived assets
- exact reproduction commands for all paper figures and tables
- assurance package contents: ODD assumptions, lineage manifest, threshold tables, verifier interface, generator provenance, and update policy for learned artifacts

### 8. Limitations, Risks, and Broader Impact

Must include:
- benchmark remains grid-based, not full 3D motion planning
- terrain is elevation-only and does not encode weather, airspace, vegetation, or dynamic obstacles
- airborne framing is advisory/surrogate-oriented, not operational validation of flight-critical autonomy
- geographic coverage limits of the source data
- potential misuse and release considerations
- intended and out-of-scope uses
- certification conclusions depend on the surrounding system, process, and regulator-approved evidence, not on the dataset alone

### 9. Conclusion

Keep short:
- what PASSAGE contributes
- why the benchmark matters
- what future benchmark extensions should add

## Suggested page budget

- Introduction: 1.0 page
- Related work and positioning: 0.75 page
- Benchmark design: 1.5 pages
- Pipeline and data schema: 1.25 pages
- Evaluation protocol: 1.25 pages
- Results and analysis: 2.25 pages
- Release, limitations, conclusion: 1.0 page

## Figure and table plan

Main paper:

- Figure 1: benchmark pipeline overview
- Figure 2: representative samples across resolutions, cost models, and AI input formulations
- Figure 3: geographic hold-out map plus tile coverage
- Figure 4: trustworthiness-oriented system diagram with learned surrogate/advisor plus verifier
- Figure 5: runtime scaling curves across solver families
- Table 1: comparison with existing pathfinding benchmarks
- Table 2: PASSAGE benchmark configuration and split counts
- Table 3: benchmark tasks mapped to supervised, graph, RL, and surrogate problem formulations
- Table 4: baseline results on core tasks and trustworthiness-oriented thresholds

Appendix:

- extra qualitative paths
- per-resolution data distributions
- additional ablations
- artifact schema examples
- checklist support material

## Experiments the paper still needs

High priority:

- a reproducible script that generates all paper tables and figures
- solver scaling benchmarks across all seven resolutions
- geographic hold-out versus random-split comparison
- diversity statistics for terrain, obstacles, path cost, and path length
- at least one learned baseline, even if simple
- a bounded airborne trustworthiness showcase use case with explicit ODD assumptions and fallback logic
- a threshold table for accuracy, bounded error, abstention, and fallback-trigger metrics
- an assurance mapping table connecting PASSAGE artifacts to current EASA / FAA expectations and future `ARP6983` concepts
- a raw-data release and licensing decision: direct redistribution versus reproducible retrieval manifests
- exact compute reporting

Medium priority:

- cross-resolution training and evaluation
- ablation on obstacle distributions
- ablation on cost models
- path-divergence analysis between free and obstacle-aware solutions
- comparison across image, graph, and surrogate-model formulations
- calibration and reliability analysis for learned components

Nice to have:

- domain expert commentary or use-inspired validation
- benchmark leaderboard protocol
- maintenance / versioning statement with future releases
- verifier-in-the-loop demonstration for a surrogate or advisory model

## Current repo issues to resolve before submission

1. Headline sample count inconsistency: the current config and dataset summaries support `14,130` default samples, while several docs and the old skeleton say `141,300`.
2. Artifact inconsistency: `outputs/dataset/summary.json` reports a large real run, while `outputs/calibrate/calibrate.json` appears to be a dummy single-tile artifact.
3. No paper-specific experiment runner: there is no single script that reproduces all paper numbers, figures, and tables end to end.
4. Learned baselines appear absent: the repo currently has solver and notebook analyses, but not paper-ready ML baselines.
5. No obvious Croissant metadata file or benchmark datasheet.
6. No `CITATION.cff` or stable archival identifier visible in the repo.
7. Source-data licensing and derivative-data release terms should be stated more explicitly in the paper and release artifacts.
8. The current narrative overuses the HEMS / “save lives” framing relative to the empirical evidence presently visible in the repo.
9. There is no clear certification demonstrator architecture yet, such as `learned surrogate + deterministic verifier + fallback`.
10. There is no assurance artifact pack yet: ODD assumptions, lineage manifest, threshold tables, monitoring logic, or update policy.
11. There is no current benchmark task definition that cleanly separates supervised, GNN, RL, and surrogate-model settings.

## Concrete additions that would materially improve selection odds

- Add one benchmark-fidelity section that proves PASSAGE is not just bigger, but harder and more informative than synthetic baselines.
- Add at least one learned image-style baseline, one graph-style baseline, and one surrogate baseline with a clean, reproducible training script.
- Add a Croissant metadata file plus a short benchmark datasheet.
- Add a one-command artifact reproduction script for all paper figures and tables.
- Add a benchmark lifecycle statement: versioning, maintenance owner, and deprecation policy.
- Add a precise release and anonymity plan for review versus post-acceptance release.
- Add a tighter license / provenance section for JAXA source tiles and derivative exports.
- Add a trustworthiness-oriented demonstrator section with a bounded airborne advisory or planner-acceleration use case.
- Add a standards-mapping table from PASSAGE artifacts to EASA guidance, FAA guidance, and the future `ARP6983` evidence model.
- Add a verifier-plus-surrogate experiment that demonstrates fallback behavior under bounded thresholds.
- Add an explicit raw-data release statement that says whether source tiles are mirrored directly or reconstructed via manifests and scripts.

## Suggested opening paragraph

Most learning-based planning and surrogate-modeling research is still evaluated on synthetic grids or game maps. Those benchmarks remain useful for algorithmic comparison, but they say little about how learned components behave on real terrain, under geographic distribution shift, with auditable provenance, or inside trustworthiness-building evidence workflows for airborne systems. PASSAGE is intended to close that gap by turning real satellite elevation data into a reproducible benchmark for terrain-aware routing, advisory learning, and surrogate modeling across scale, cost models, split regimes, and fully transparent release artifacts covering both the data and the generator used to create them.

## Suggested contribution paragraph

This paper does not introduce a new flight-control algorithm. Instead, it contributes a benchmark and generation pipeline: a multi-resolution real-terrain task suite derived from auditable source data, explicit benchmark protocols for scaling, generalization, and surrogate modeling, rich sample-level metadata and export artifacts, and a release plan designed for reproducible evaluation of bounded learned airborne components. PASSAGE is shared with the aim of helping build trustworthiness toward the levels of assurance expected by EASA and the FAA, as future `ARP6983`-style guidance becomes concrete, while enabling pioneers who want to embed AI onboard aircraft and in other safety-critical applications. Its value should therefore be judged by benchmark utility, coverage, rigor, transparency, and reusability.

## Raw data release note

- The paper should promise release of the full derived benchmark, manifests, metadata, preprocessing code, and generator logic.
- If third-party terms allow direct redistribution of source tiles, say so precisely and archive the raw package.
- If third-party terms do not clearly allow mirroring, release deterministic source-data manifests and downloader scripts instead of mirroring the raw tiles.
- Conservative wording for the paper:
  - `We release the full derived benchmark and the complete source-data retrieval and preprocessing pipeline.`
  - `Where direct redistribution of third-party source data is not appropriate, we release reproducible manifests and scripts that rebuild the raw-data stage without altering the benchmark protocol.`
