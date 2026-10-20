# PASSAGE Datasheet

*Table 1. Document revision.*

| Author | Date | Version | Status |
| --- | --- | --- | --- |
| Geoffrey Delhomme | 2026-10-19 | 1.0 | Generated, verified and admitted for publication; not yet uploaded |

This datasheet follows the questions of *Datasheets for Datasets*. It describes
PASSAGE v1.0 (Hub tag `v1.0`), published at
[huggingface.co/datasets/thalesgroup/passage](https://huggingface.co/datasets/thalesgroup/passage),
and accompanies its [Croissant metadata](https://huggingface.co/datasets/thalesgroup/passage/blob/main/croissant.json)
(Croissant 1.1 with Responsible AI fields). The same file is kept in the
[code repository](https://github.com/ThalesGroup/passage).

The release receipts summarise PASSAGE v1.0 as follows:

PASSAGE v1.0 contains 4,400,000 core scenes and 26,400,000 objective/obstacle variants; 1,792 separate preview rows do not add independent training observations. Terrain and labels use persisted float32 inputs decoded before authoritative float64 graph solves. Missing/nodata source crops are rejected. The qualified outcomes include 26,358,039 successful exact routes and 41,961 independently proved no-path variants. No learned model has been trained or evaluated on PASSAGE v1.0.

The paper's learned-model results are historical results of the original study, reported as obtained; not re-run on the released dataset. This datasheet documents a computational
annotation and qualification process. It is not an operational deployment
certification.

## Motivation

### For what purpose was the dataset created?

PASSAGE supports reproducible, verifiable research on constrained path
planning over real terrain. It provides multi-resolution scenes from a declared
Operational Design Domain (ODD), three terrain cost models, procedural
obstacles, and an exact reference outcome for every route variant, so that
classical and learned planners can be compared on legality, cost and failure.

### Who created this dataset and on behalf of which entity?

The dataset was created by Geoffrey Delhomme, François Trap, and Fateh Kaakai on behalf of Thales Group research. See `CITATION.cff` for full attribution. Maintenance is owned by the Thales Group open-source research SIG; community contact via GitHub Issues, Discussions, or `oss@thalesgroup.com`.

### Who funded the creation of the dataset?

Internal research funding (Thales Group). No external grant is associated with this release.

## Composition

### What do the instances represent?

Each core row is one scene: an N×N crop of the JAXA ALOS AW3D30 digital
surface model, a start cell, a goal cell, a binary obstacle mask, structured
metadata, and six reference route variants. The variants combine three
objectives (`elevation`, `slope_uphill`, `energy`; Parquet solver names
`astar_elevation`, `astar_slope`, `astar_energy`) with free terrain and with
obstacles blocked.

### How many instances are there?

4,400,000 core scenes at seven resolutions: 1,000,000 each at 64, 128, 256 and
512; 250,000 at 1024; 100,000 at 2048; 50,000 at 4096. Each resolution follows
the 90/2/4/4 train/calibration/validation/test allocation. The scenes carry
26,400,000 route variants: 26,358,039 exact routes and 41,961 proved no-path
outcomes, with none unresolved. Seven `_samples` configurations add 1,792
preview rows (64 per split and resolution); they repeat admitted scenes and do
not enlarge the training population. The data occupy about 1.27 TB of Parquet
in 5,997 shards.

*Table 2. Core scenes per resolution and split.*

| Resolution | train | calibration | validation | test | Total | Preview rows per split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64×64 | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 128×128 | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 256×256 | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 512×512 | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 1024×1024 | 225,000 | 5,000 | 10,000 | 10,000 | 250,000 | 64 |
| 2048×2048 | 90,000 | 2,000 | 4,000 | 4,000 | 100,000 | 64 |
| 4096×4096 | 45,000 | 1,000 | 2,000 | 2,000 | 50,000 | 64 |
| **All** | 3,960,000 | 88,000 | 176,000 | 176,000 | 4,400,000 | 1,792 rows in total |

### Does the dataset contain all possible instances or is it a sample?

It is a seeded sample of a configured parameter space, not an enumeration.
Each scene derives its seed as `blake2b(seed, resolution, index, split)` with
generation seed 42 and split seed 42. Regeneration with other settings creates
a separately identified version.

### What data does each instance consist of?

- `id`: zero-padded scene index, unique within a configuration and split.
- `tensor`: zstd-compressed float32 array of shape (N, N, 3). Channel 0 is
  elevation normalised with the global calibration; channel 1 holds markers
  (−1 background, 0 start, 1 goal); channel 2 is the binary obstacle mask.
- `metadata`: split, calibration, source and overlapping tiles, crop bounds,
  markers, obstacle statistics, seeds, provenance identifiers, solver and graph
  parameters, and per-variant status, cost, length and timing.
- `metadata_json` and `scene_receipt`: complete metadata and the scene receipt
  as JSON text, including source windows, source hashes and attempts.
- `path_<solver>_free`, `path_<solver>_obstacles`: ordered `[row, col]` cells
  from start to goal.
- `_samples` configurations add one rendered preview image per objective. The
  rendering is not an elevation measurement.

### Is there a label or target associated with each instance?

Yes. Each route variant has a terminal status. `ok` routes are exact optima
under the frozen graph specification below, computed in float64 from the
decoded float32 tensor. `infeasible` means the reference proved that no path
exists; the route is empty and the cost is null. Tied optima can have different
geometry. Successful routes are also listed in separate supervision rosters.

### Is any information missing from individual instances?

No admitted scene is incomplete. Crops that touch missing tiles, nodata cells,
incomplete windows or elevations outside the frozen calibration range are
rejected rather than padded or clipped. Each scene allows up to 100 attempts and
records rejected attempts in its receipt. All requested scenes completed.

### Are relationships between individual instances made explicit?

Yes. Each scene receipt records its source windows (tiles, offsets and
hashes), so overlap and shared terrain can be computed. The six variants of a
scene share terrain and endpoints; scenes from the same 5°×5° block, and from
different resolutions, can share source tiles.

### Are there recommended data splits?

Yes. Every source tile belongs to one split. Tiles inside the South America
hold-out mask go to `test`; the others go to `train`, `calibration` or
`validation` by seeded 5°×5° blocks. A crop is admitted only if all its source
tiles and a one-tile guard ring belong to its split, so source-cell windows of
different splits are disjoint. Fix a method's free settings on the
`calibration` split and freeze them before the single test pass. The frozen
policy is:

```json
{
  "block_degrees": 5,
  "buffer_tiles": 1,
  "holdout_mask_hash": "b35306e3f32969c10608e16f556fe2038930b0496482ca9d4a8a7f5de3350ab0",
  "ratios": {
    "calibration": 0.02,
    "test": 0.04,
    "train": 0.9,
    "validation": 0.04
  },
  "seed": 42,
  "version": "source-tile-blocks-v1"
}
```

### Are there any errors, sources of noise, or redundancies?

Elevation inherits the limitations of the AW3D30 digital surface model,
including canopy and building surfaces and possible sensor artefacts.
Obstacles are abstractions, not measured hazards. Labels are exact only under
the stated graph contract. Release verification checked scene counts, terminal
statuses and cohort and supervision hashes for every resolution. A bounded
replay of 56 selected scenes reproduced 84 scientific records and 84 preview
images byte for byte; it does not certify every row or cross-platform
reproducibility.

*Table 3. Proved no-path variants per resolution.*

| Resolution | Route variants | Exact routes | Proved no-path |
| --- | ---: | ---: | ---: |
| 64×64 | 6,000,000 | 5,987,736 | 12,264 |
| 128×128 | 6,000,000 | 5,989,788 | 10,212 |
| 256×256 | 6,000,000 | 5,991,657 | 8,343 |
| 512×512 | 6,000,000 | 5,992,002 | 7,998 |
| 1024×1024 | 1,500,000 | 1,498,014 | 1,986 |
| 2048×2048 | 600,000 | 599,250 | 750 |
| 4096×4096 | 300,000 | 299,592 | 408 |
| **All** | 26,400,000 | 26,358,039 | 41,961 |

### Is the dataset self-contained, or does it link to other resources?

The Parquet configurations contain the tensors, routes and metadata needed to
use the data. Raw AW3D30 tiles are not redistributed; scene receipts identify
the exact source windows and hashes. `release-receipts/` archives the resolved
configuration, the producer source code, generation provenance, cohort and
supervision rosters, and verification results.

### Does the dataset contain confidential, personally identifiable, sensitive, or offensive information?

No. The data contain public elevation data, procedural obstacles and markers,
and computed routes. There are no people, personal data or biometric data.
Local file paths, user names and host names were redacted from the published receipts, calibration files, producer source archive, executed notebooks and one Parquet scene receipt; `release-receipts/privacy-redactions.json` lists every changed file with its original and published SHA-256.

## Collection process

### How was the data acquired?

The pipeline downloads AW3D30 archives packaged in 5°×5° regions and uses
their 1°×1° GeoTIFF tiles. The local inventory matches all 23,993 tiles of
JAXA's official AW3D30 list (checked on 11 October 2026), and
`calibration/calibrate.json` pins the filename, size and SHA-256 of every tile.
Tiles are normalised to 3600×3600 cells by nearest angular-centre sampling. The
complete calibration range is **-430.0 m to 8767.0 m**.

### What mechanisms or procedures were used to collect the data?

A code-driven pipeline (`passage download`, `passage calibrate`,
`passage regenerate` in the code repository). For each scene, the generator
draws a centre tile among the tiles eligible for its split and a target
elevation uniformly over the calibration range. It takes the cell nearest the
target as the start, places the crop around it, and draws the goal uniformly
among the other cells. No crowdsourcing, human annotators or subjects.

### Was any preprocessing, cleaning, or labeling done?

- Elevation is normalised with the global calibration range.
- Obstacles are super-ellipses with a coverage target up to 30 % of the crop,
  size ratios 0.02 to 0.25 and exponents from 1 to 80 (log-uniform), with up to
  50 mask attempts and 200 candidates per attempt. They avoid both markers.
- An exact float64 A\* reference (`src/passage/reference.py`) solves the six
  variants from the decoded tensor, with no expansion budget. The graph and its
  weights are:

```json
{
  "astar_elevation": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "elevation",
    "cost_weight": 62.4,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  },
  "astar_energy": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "energy",
    "cost_weight": 1125.3,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  },
  "astar_slope": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "slope_uphill",
    "cost_weight": 8886.2,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  }
}
```

  The graph is eight-connected, with 30 m orthogonal and 30√2 m diagonal edges.
  A diagonal step is checked against its destination cell only. For a step
  from u to v, with d the edge length and h the elevation in metres:
  `elevation` costs d · (1 + w · (h_v − h_min) / (h_max − h_min));
  `slope_uphill` costs d · (1 + w · max(0, h_v − h_u) / (h_max − h_min));
  `energy` costs d + w · max(0, h_v − h_u). h_min and h_max are the crop's
  extrema.

### Was the "raw" data saved in addition to the preprocessed data?

The AW3D30 tiles and the calibration cache are kept locally by the producers,
not published. Scene receipts and `calibration/calibrate.json` identify every
source byte used.

### Is the software available for the preprocessing?

Yes. The generator, reference solver and evaluator are in
[github.com/ThalesGroup/passage](https://github.com/ThalesGroup/passage)
(`src/passage/`, `src/passage-eval/`). The producer source used for this
dataset is archived in `release-receipts/producer-source.zip`. One host label in
that archive was redacted (see `release-receipts/privacy-redactions.json`);
`release-receipts/generation-provenance.json` keeps the original file hashes.

## Uses

### Has the dataset been used for any tasks already?

No learned model has been trained or evaluated on PASSAGE v1.0. The paper's
learned results (dense path-mask segmentation and cost-to-go regression) are
historical results of the original study, reported as obtained;
not re-run on the released dataset.

### What (other) tasks could the dataset be used for?

- Benchmarking classical planners (Dijkstra, A\*, Theta\*, D\*, JPS) on real terrain.
- Training and evaluating learned routing surrogates: vision, graph and
  reinforcement-learning planners.
- Studying resolution scaling, geographic transfer and cost-model transfer.

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?

- Scenes are static 2D grids; they do not support 3D motion planning or
  dynamic environments.
- Obstacles and endpoints are synthetic; they do not reflect real hazard or
  mission distributions.
- Starts favour the elevation extremes of their tile, because the target
  elevation is drawn over the global range.
- Centre tiles are drawn uniformly among eligible tiles, so coverage is not
  proportional to ground area.
- The test split covers South America only; test scores measure transfer to
  that region.

### Are there tasks for which the dataset should not be used?

- Operational flight planning or search-and-rescue dispatch. PASSAGE is a
  research benchmark and does not replace EASA or FAA certification processes.
- Conclusions about weather, airspace, vegetation, urban structures or
  regulation.
- Safety-of-life decisions without an in-the-loop deterministic verifier and
  fallback solver.

## Distribution

### Will the dataset be distributed to third parties?

Yes, publicly, at
[huggingface.co/datasets/thalesgroup/passage](https://huggingface.co/datasets/thalesgroup/passage).
Use the release tag `v1.0`.

### How will it be distributed?

As Parquet shards, one folder per configuration and split, with the dataset
card, this datasheet, `croissant.json`, `CITATION.cff`, `LICENSE`,
`DATASET_LICENSE.md`, release receipts, the calibration file and executed
notebooks.

### When will it be distributed?

With the camera-ready version of the NeurIPS 2026 paper. The release receipts
were sealed before upload.

### Will the dataset be distributed under a license? Will it have an associated DOI?

- Code: MIT (`LICENSE`).
- Data: derived from JAXA AW3D30 under the
  [JAXA terms of use](https://earth.jaxa.jp/en/data/policy/), summarised in
  `DATASET_LICENSE.md`. The data are neither MIT-licensed nor public domain.
- No DOI is assigned yet.

### Have any third parties imposed IP-based or other restrictions?

JAXA's terms apply to the derived terrain: credit JAXA and the product, keep
third-party credits, and notify JAXA before commercial use. Credit line:
**The original data used for this product have been supplied by JAXA's ALOS World 3D–30m (AW3D30).**

AW3D30 fills voids with third-party elevation data. JAXA's [AW3D30 Version 4.1 Product Description](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/data/aw3d30v4.1_product_e_1.0.pdf) (checked on 19 October 2026; Table 1, mask file; Section 4, Related URLs) lists GSI Digital Topographic Map 5 m and 10 m mesh data (Geospatial Information Authority of Japan), SRTM-1 v3 (NASA/JPL), Viewfinder Panoramas DEM (Jonathan de Ferranti), ASTER GDEM v2 and v3 (NASA/METI), ArcticDEM v2 and v3 (NGA/NSF) and ArcticDEM v4, TanDEM-X 90m DEM (DLR), REMA v1.1 (PGC, University of Minnesota) and Copernicus DEM GLO-30 (ESA). For the Copernicus DEM it states: "AW3D30 is produced using Copernicus DEM GLO-30-F © DLR e.V. 2014-2018 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA, all rights reserved". Retain these credits in derived works.

### Do any export controls or other regulatory restrictions apply?

None are known. AW3D30 is a public global elevation product.

## Maintenance

### Who is supporting / hosting / maintaining the dataset?

Thales Group open-source and critical AI research contributors. Contact: `oss@thalesgroup.com`; issues and discussions on the GitHub repository. Hosting: Hugging Face.

### How can the owner be contacted?

GitHub Issues and Discussions at <https://github.com/ThalesGroup/passage>, or email `oss@thalesgroup.com`.

### Is there an erratum?

Errata will be listed in the code repository's `CHANGELOG.md` and on the
dataset card.

### Will the dataset be updated?

Any semantic change creates a new version (new Hub tag) with regenerated receipts,
an updated card and Croissant file, and a `CHANGELOG.md` entry, staged and
reviewed separately before upload.

### If the dataset relates to people: are there applicable limits on the retention of the data associated with the instances?

Not applicable.

### Will older versions of the dataset continue to be supported/hosted/maintained?

PASSAGE v1.0 is frozen by the Hub tag `v1.0`. A later version would be released
under a new tag; v1.0 stays available under its tag.

### If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?

Yes, through pull requests and issues on the code repository (see
`CONTRIBUTING.md`). A change to data semantics requires a new version and
updated Croissant and datasheet sections.

## Lifecycle statement

- **Status**: active.
- **Versioning**: version 1.0, frozen by the Hub tag `v1.0`; `version` is 1.0.0
  in `croissant.json` and 1.0 in `CITATION.cff`. Receipts and Parquet rows also
  carry an internal `dataset_version` field.
- **Maintenance owner**: Thales Group open-source research SIG; issues/PRs on GitHub.
- **Deprecation policy**: deprecations are announced in `CHANGELOG.md` and on
  the dataset card.
- **Security contact**: see `SECURITY.md` in the code repository.
