# Data qualification trace

Table 1. Document revision.

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

## 1. Purpose and scope

PASSAGE owns the terrain sources, generated data, labels, release receipts and data
qualification. This guide follows the prerequisites in order: authenticate terrain,
freeze calibration and provenance, generate and export, then qualify a frozen cohort
and complete the release gates. Training and model experiments belong to the
companion vision repository; PASSAGE supplies their data contract.

A **cohort** is an explicitly frozen set of scenes and route variants with identified
source shards, splits and qualification outcomes. A successful pilot or bounded
replay establishes evidence for its stated scope. Complete release admission requires
all requested scenes, all seven resolutions and the final publication checks.

## 2. Audit the terrain inventory before freezing inputs

Match the local inventory to the official Japan Aerospace Exploration Agency (JAXA)
AW3D30 catalogue. Record the catalogue bytes, retrieval provenance and digest; check
unique tile IDs, missing/extra/duplicate files and readable GeoTIFF metadata, including
band count, data type, coordinate system, dimensions and one-degree bounds.

The 2026-10-11 local audit recorded **23,993 official IDs and 23,993 valid DSM files**,
with no missing, extra or duplicate IDs and no calibration filename/size differences.
Its evidence is under `outputs/regeneration/jaxa-inventory-verification-20261011/`:
`audit-report.json`, `catalogue-source.json` and `summary.md`. This was a metadata
check: no live pixel checksum campaign or pixel-integrity decode was performed.
Stored archive receipts cover their recorded members only.

The proposed final-download verification hook remains an **unapplied patch** in that
evidence directory. Keep active producer source, configuration and calibration frozen
while regeneration runs. Apply and validate any downloader change at a safe checkpoint,
then record its new identity. See the existing [downloader](../src/passage/download.py).

## 3. Complete calibration and freeze the scientific identity

Run full calibration on the admitted inventory before production generation. Require a
complete receipt, all expected tiles processed, zero unexplained failures or exclusions,
and validated inventory and receipt digests. Preserve the exact calibration bytes and
SHA256 pin throughout generation, export, replay and consumer handoff.

The inspected active calibration records 23,993/23,993 processed tiles, zero failed or
excluded tiles, and global elevations from −430 to 8,767 metres. Its pinned SHA256 is
`996ba3c363693a6b45dc7431d7b0a5e4d833f4f4c45f51e628478734ad4bd087`.
Calibration coverage alone does not prove official download coverage; retain Section 2's
independent inventory audit. See [calibration validation](../src/passage/calibrate.py).

Freeze resolved configuration, producer code and source archive, dependency lock and
runtime versions, raw inventory, holdout mask, generation/split seeds, per-scene seeds,
source windows, normalization and no-data policy, graph/objective identities and tie
rules. Preserve `release-receipts/calibration.json`, `resolved-config.yaml`,
`raw-inventory.json`, `code-inventory.json`, `producer-source.zip` and
`generation-provenance.json`. See [release orchestration](../src/passage/regenerate.py).

## 4. Generate every requested scene transactionally

Use the verified [sequential recipe](../README.md#-sequential-full-dataset-generation-and-structural-analysis)
from the repository root. The real CLI supports inspection with
`uv run --frozen passage regenerate --dry-run` and isolated pilots with `--pilot`.
Keep pilots separately identified; their counts cannot establish full completeness.

Production covers **64, 128, 256, 512, 1024, 2048 and 4096** pixels per side and four
splits: train, calibration, validation and test. The default plan requests 4.4 million
core scenes. Core configurations contain tensors and metadata without images;
`_samples` configurations contain preview images, normally 64 scenes per split.

Commit scene artifacts and terminal receipts together. Preserve exact success, proved
no-path, censored search, numerical failure and runtime failure separately. Operational
retries retain their history; scientific rejection attempts and accepted seeds remain
part of reproducibility evidence. Solve labels from the decoded persisted float32
terrain under the recorded native graph. See [release generation](../src/passage/successor_generation.py).

## 5. Verify export before removing owned scratch

For each committed batch, verify shard hashes, schemas, row counts, IDs, receipt
bindings, core/preview separation and provenance chains. Check that every requested
index has one terminal outcome, every admitted complete scene appears in core data,
preview counts match the plan, and no extra or missing Parquet shard exists.

Only remove raw scratch artifacts owned by a verified exported batch. Keep final
Parquet shards, scene/batch receipts, source data and calibration intact. Persist
workflow and generation-stage receipts so resume can distinguish exported data from
unfinished work. Final verification writes `release-verification.json`,
`cohort-<resolution>.jsonl` and `supervision-<resolution>.jsonl`; the latter contains
successful exact route variants. Proved no-path outcomes remain identifiable in the
cohort. Censored, numerical or runtime label failures block release admission.
These checks and cleanup ownership rules are implemented in
[regeneration](../src/passage/regenerate.py) and [export](../src/passage/export.py).

## 6. Qualify the frozen cohort and replay native artifacts

Bind the cohort to exact shard, receipt, configuration, calibration and producer hashes.
Complete a census of every core row in its declared scope: IDs and splits, native graph
and objective identities, label terminal statuses, calibration bindings and source-cell
footprints. Check cross-split source support and recorded windows; a tile-name or
bounding-box diagnostic alone cannot prove source-cell independence.

Independently check route legality, endpoints, obstacle/corner rules, stored cost and
exact reference status. Qualification requires legality, stored-cost agreement and an
authenticated exact optimum. Preserve no-path and unresolved findings explicitly.
For the consumer census, population checks authenticate the exact producer
solver certificate and verify stored routes and costs. They do **not** independently
re-solve every graph. The independent native replay below is bounded; record its
sample count separately from the full census count.
Freeze deterministic native replay selections of **eight scenes per split per resolution**
for the cohort gate; regenerate tensors, metadata, source windows, seeds and all route
variants and compare their recorded scientific values and hashes. Replay preview PNG
bytes separately. Never fill absent labels, invent success or replace missing evidence
with a synthetic receipt. A scene that cannot be replayed leaves the declared gate pending.

The existing `uv run --frozen passage qualify --help` describes authenticated snapshot
phases (`authenticate`, `census`, `routes`) and requires an immutable dataset revision.
Its stratified diagnostic route roster is a separate protocol, not the eight-per-split
replay gate or a population-wide route check. See
[metadata census](../src/passage/qualification.py),
[route audit](../src/passage/qualification_runner.py) and
[receipt reporting](../src/passage/qualification_reports.py).

## 7. Complete release gates and hand off data evidence

On 2026-10-12, the producer's full export verification reported all **4,400,000
scenes** at the seven planned resolutions, with **26,358,039 successful route
variants**, **41,961 proved infeasible variants**, and zero unresolved variants.
The [observed export certificate](../outputs/regeneration/export-verification-observed-20261012.json)
records the counts, cohort and supervision identities, and calibration/configuration
pins. These are producer verification results, not independent recomputation of
every label or final publication admission.

An independent read of the Parquet footers also confirmed **64 examples per
split in all seven `_samples` configurations**, or 1,792 example rows. The
[sample inventory](../outputs/regeneration/sample-inventory-observed-20261012.json)
records counts and image-column schemas. This inventory does not certify every
image payload; scientific and PNG replay remain separate gates.

After full generation/export, require qualified admission across all seven resolutions,
then execute and validate the final analysis notebooks and publication metadata.
`ready-upload.json` must record the resulting admission; upload remains a separate
authorized action. A notebook smoke run, pilot, partial export or successful replay
cannot replace this gate. See [publication validation](../src/passage/notebook_export.py).

On 2026-10-13, a separately authorized recovery admitted all **42 publication
notebooks** and the derived publication metadata. Its persistent parent observed
the finalizer's actual exit code 0. The [completion receipt](../outputs/regeneration/metadata-recovery-v3-20261013/completion.json)
has SHA256 `6c255431339aef6305b4f862726d5aaafcea22fa685df96b679e0065cd41e2a8`.
The recovery preserved scientific generation sources and identities, calibration,
configuration, all 5,997 Parquet shards and the completed notebook outputs.
Earlier failed producer and recovery records, including the first journal and
readiness receipt, remain unchanged. The versioned completion receipt and
the separately pinned [audit-only gate authorization](../outputs/regeneration/final-audit-gate-v4-20261013/authorization.json)
authorize the subsequent scientific and PNG audits;
metadata completion alone does not establish their success.

This release's final metadata requires the archived, pinned **feature-order view**.
It orders the existing feature descriptors to match physical Parquet columns,
preserving their names, types and nullability. The strict native Arrow schema
comparison remains enabled. The [recovery limitation](../outputs/regeneration/metadata-recovery-v3-20261013/ORDERED_FEATURE_VIEW.md)
and [source archive](../outputs/regeneration/metadata-recovery-v3-20261013/metadata-execution-source-v3.zip)
describe the supplement. The unmodified native CLI still has the column-order
bug under the current source freeze and cannot reproduce this final metadata
alone. A permanent fix belongs in a future unfrozen revision with its own
source identity, after a safe source handover.

As inspected on 2026-10-11, full regeneration was running and `ready-upload.json`
reported `verification_pending`. A 64-scene-per-resolution pilot and a separate
four-scene-per-resolution replay were verified. The earlier native scientific proof
covered 24 distinct scenes and 144 route variants at 64/128/256 across all four splits;
the corresponding bounded scientific and preview-image replays both reported
`passed`. Their records remain available alongside the later final proof.
The old `reproducibility-current-proof.json` predates the completed PNG check;
read its associated `audit-state.json` and `preview-image-replay/preview-state.json` under
`<scratch-root>/reproducibility-audit-current-20261011/` together.

On 2026-10-13, the final frozen scientific auditor and PNG auditor both exited
with code 0. All **84 scientific comparison records** passed with empty mismatch
lists, covering **56 distinct selected scenes** across all seven resolutions and
four splits. All **84 PNG comparisons** were byte-identical. The
[final proof](../outputs/regeneration/reproducibility-final-proof-recovery-20261013.json)
pins the [combined passed gate](../outputs/regeneration/reproducibility-final-state.json),
both audit receipts, original auditor sources, separately authorized recovery and
preserved earlier failures. The audit receipts are `audit-state.json` and
`preview-image-replay/preview-state.json` under
`<scratch-root>/reproducibility-audit-full-recovery-20261013/`.

The selection takes the first row of the first and last committed core shards
in each split, plus the first preview row. Overlap between core and preview rows
leaves 56 distinct scenes. This is a **bounded selected-scene reproducibility
proof**: it does not independently recompute every label in the 4.4-million-scene
release, certify every earlier failure episode or establish cross-platform
reproducibility. The separate `passage qualify` report has not been run on this
release, and upload remains a separate authorized action.

On 2026-10-18, a [fresh terminal-artifact check](../outputs/regeneration/terminal-observation-20261018.json)
confirmed the completion pin, all 42 admitted notebook hashes (including
vision notebook 09), the eight publication-metadata hashes, and both final
seven-resolution audit-state pins. All matched. This was a read-only check
of completed artifacts, not new generation, another full shard scan or an
independent recomputation of every route. No upload was performed.

The active consumer qualification campaign is currently implemented in vision. Its
`qualification.json`, `roster.json`, `source-census.jsonl` and `progress.json` must bind
the full declared cohort and native checks; its partial progress is not a completed
census. This implementation location does not transfer training logic into PASSAGE.

The completed 2026-10-12 census for **256×256, elevation, obstacles** covered
1,000,000 source scenes and qualified 997,219 for successful-path supervision.
Its sealed cohort SHA256 is
`006c606fe7e8c96c5cac894616000dc59bf83bf2b44e1154703405db2795b123`.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 900,000 | 897,518 |
| Validation | 40,000 | 39,884 |
| Calibration | 20,000 | 19,932 |
| Test | 40,000 | 39,885 |

The excluded 2,781 scenes remain in the source census; no replacements are drawn.
This population certificate is separate from the 32 independent native replays
(eight per split), regression-target completion and GPU training admission.

The R256 **energy** and **slope** objectives have separate sealed cohorts:
`cbf0610cf285baf19ee55004cc5eddad49f35afd88b874bbd972853765c1d361`
and `74cfec48acd2c1af99ae123d09c8ddc7dc9fd35a959269d77736b13690b73fc8`,
respectively. Their [energy handoff](../outputs/regeneration/cohort-qualification-r256-energy-20261014.json)
and [slope handoff](../outputs/regeneration/cohort-qualification-r256-slope-20261014.json)
record the 2026-10-14 verification of each qualification, roster, full source
census, qualified records, split indices and producer batch receipts. Both have
the same source and qualified split counts shown above, with 2,781 selected
infeasible variants retained without replacement. Their graph, cost and label
identities remain objective-specific; equal population counts do not make their
admissions interchangeable.

Each objective's certificate records zero cross-split source-footprint violations
and bounded independent native re-solving of eight scenes per split. The handoff
verification recounts the complete population and checks the sealed auxiliary
files; it does not rehash Parquet payloads or independently re-solve every label.
Regression target/scale verification and model resource, resume and backup
admission remain separate gates.

The **64×64, elevation, obstacles** census completed on 2026-10-12 with
1,000,000 source scenes and **995,912 qualified examples**. The
[data receipt](../outputs/regeneration/cohort-qualification-r64-20261012.json)
binds the qualification and roster hashes, cohort identity and native replay.
The recorded source-footprint population had zero cross-split partition violations;
independent exact-reference re-solving remained bounded to eight scenes per split.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 900,000 | 896,325 |
| Validation | 40,000 | 39,845 |
| Calibration | 20,000 | 19,918 |
| Test | 40,000 | 39,824 |

The 4,088 exclusions are retained without replacement. This data admission does
not establish regression-target completion, model training admission or full-release
publication readiness.

The **128×128, elevation, obstacles** census completed on 2026-10-12 with
1,000,000 source scenes and **996,596 qualified examples**. Its
[data receipt](../outputs/regeneration/cohort-qualification-r128-20261012.json)
binds the sealed roster and qualification hashes. All source footprints were
recorded with zero cross-split partition violations; independent replay remains
bounded to eight scenes per split.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 900,000 | 896,916 |
| Validation | 40,000 | 39,869 |
| Calibration | 20,000 | 19,931 |
| Test | 40,000 | 39,880 |

The 3,404 exclusions remain in the source census without replacement. Vision's
private regression-target construction completed for the **936,785 training and
validation examples**, with a sealed manifest and training-derived scale. Model
resource/recovery qualification remains a separate gate.

The **1024×1024, elevation, obstacles** census completed on 2026-10-12 with
250,000 source scenes and **249,338 qualified examples**. Its
[data receipt](../outputs/regeneration/cohort-qualification-r1024-20261012.json)
binds the sealed roster and qualification hashes. All 250,000 source footprints
were recorded with zero cross-split partition violations; independent replay
remains bounded to eight scenes per split.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 225,000 | 224,409 |
| Validation | 10,000 | 9,973 |
| Calibration | 5,000 | 4,984 |
| Test | 10,000 | 9,972 |

The 662 exclusions remain in the source census without replacement. Regression
targets and model resource/recovery qualification are separate pending gates.

The **512×512, elevation, obstacles** census completed on 2026-10-12 with
1,000,000 source scenes and **997,334 qualified examples**. The
[data receipt](../outputs/regeneration/cohort-qualification-r512-20261012.json)
binds the complete census, roster, split indices and qualification hashes,
reverified at handoff. All source footprints were recorded with zero cross-split
partition violations. Independent exact-reference replay covers eight scenes
per split; it does not independently recompute every source label.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 900,000 | 897,624 |
| Validation | 40,000 | 39,883 |
| Calibration | 20,000 | 19,942 |
| Test | 40,000 | 39,885 |

The 2,666 exclusions remain in the source census without replacement. This
receipt covers elevation with obstacles; other conditions, regression targets,
model resource/recovery admission and final publication replay remain separate.

The **2048×2048, elevation, obstacles** census completed on 2026-10-12 with
100,000 source scenes and **99,750 qualified examples**. The
[data receipt](../outputs/regeneration/cohort-qualification-r2048-20261012.json)
binds the qualification, roster and complete source-census hashes. All source
footprints were recorded with zero cross-split partition violations. Independent
exact-reference replay remains bounded to eight scenes per split.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 90,000 | 89,778 |
| Validation | 4,000 | 3,987 |
| Calibration | 2,000 | 1,998 |
| Test | 4,000 | 3,987 |

The remaining **250 selected variants are infeasible**, rather than successful-path
supervision. Their outcomes remain in the source census without replacement.
The full census and sealed roster files, split indices and hashes were verified at
handoff. Regression targets and chosen-MIG resource/recovery admission remain
separate requirements. The full seven-resolution publication replay independently
passed as recorded above; this census alone did not establish that result.

The **4096×4096, elevation, obstacles** census completed on 2026-10-13 with
50,000 source scenes and **49,864 qualified examples**. The
[data handoff receipt](../outputs/regeneration/cohort-qualification-r4096-20261014.json)
records the 2026-10-14 verification of the sealed qualification, roster, source
census, split indices and producer batch receipts. All 50,000 source footprints
were recorded with zero cross-split partition violations. The successful
population uses authenticated native producer certificates; independent
exact-reference re-solving remains bounded to eight scenes per split.

Table 8. Source and qualified scene counts for the 4096×4096 elevation cohort
with obstacles.

| Split | Source scenes | Qualified examples |
| --- | ---: | ---: |
| Training | 45,000 | 44,873 |
| Validation | 2,000 | 1,997 |
| Calibration | 1,000 | 999 |
| Test | 2,000 | 1,995 |

The remaining **136 selected variants are infeasible** and remain in the census
without replacement. The handoff rechecked all source and qualified record
counts, split-index membership and the successful census process exit. It did
not rehash the Parquet payloads or independently recompute every label.
Model fit, resume and backup qualification remain separate gates; this receipt
does not admit or launch an R4096 GPU fit.

A completed resolution can be admitted to training before the other resolutions finish,
using its sealed export, complete consumer census, split independence and bounded
native replay. That admission applies only to the declared resolution, objective and
training roster. Regression additionally requires verified physical targets and a scale
derived from training data alone. The seven-resolution scientific and PNG replay
required for publication of the full release passed on 2026-10-13; retain its
bounded selection scope when citing that evidence.

Hand off the frozen cohort and successful-supervision rosters, calibration pin, graph
contracts, shard/receipt hashes, qualifications, exclusions and replay results to vision.
They define which data a consumer may use and how to identify it. Training schedules,
checkpoints and learning curves are traced in vision; data qualification alone makes
no claim about model performance or completed training.
