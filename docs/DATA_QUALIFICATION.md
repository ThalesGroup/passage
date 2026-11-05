# Data qualification trace

Table 1. Document revision.

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-21 | 1.1 |

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

Use the [release-generation guide](guide/export.md#all-in-one-passage-regenerate)
from the repository root. The real CLI supports inspection with
`uv run --frozen passage regenerate --dry-run` and isolated pilots with `--pilot`.
Keep pilots separately identified; their counts cannot establish full completeness.

Production covers **64, 128, 256, 512, 1024, 2048 and 4096** pixels per side and four
splits: train, calibration, validation and test. The default plan requests 4.4 million
core scenes. Core configurations contain tensors and metadata without images;
`_samples` configurations contain preview images, normally 64 scenes per split.

Commit scene artifacts and terminal receipts together. Preserve exact success,
censored search, numerical failure and runtime failure separately. Start and goal cells
that the obstacles disconnect are rejected as inputs before any label is solved
(`passage.reference.endpoints_connected`, on the reference graph); a solver proof that
contradicts this check fails the scene. Operational retries retain their history;
scientific rejection attempts and accepted seeds remain part of reproducibility
evidence. Solve labels from the decoded persisted float32 terrain under the recorded
native graph. See [release generation](../src/passage/successor_generation.py).

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
successful exact route variants, which in PASSAGE v1.0 are all the variants of the
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
authenticated exact optimum. Preserve unresolved findings explicitly.
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

Release verification reruns every batch manifest of the seven resolutions: shard
hashes and schemas, row identities, scene receipts, the chain between batch
manifests, and the cohort and supervision rosters. For PASSAGE v1.0 it reports all
**4,400,000 scenes** and **26,400,000 exact route variants**, with zero unresolved
variants and none without a path, because disconnected endpoints are rejected at
generation (Section 4). `release-receipts/release-verification.json` records the
counts, roster identities and calibration/configuration pins;
`publication-facts.json` and `publication-metadata.json` bind the published documents
to them. Each `_samples` configuration holds 64 preview rows per split, 1,792 rows
in all.

`release-receipts/connectivity-repair-v1.json` lists the 13,987 scenes whose stored
input comes from the disconnected-endpoint rule of Section 4, applied after the first
export. For each scene it records the rejected and accepted attempts and the old and
new receipt hashes, and it lists every rewritten shard and batch manifest. Untouched
control scenes regenerated by the same generator reproduce their published rows, and
the generator's source is archived beside the receipt.

A bounded native replay of 56 selected scenes reproduced 84 scientific records and
84 preview images byte for byte. It does not certify every row or cross-platform
reproducibility.

The consumer freezes each training cohort from the supervision rosters, per
resolution, objective and obstacle condition. Every variant is exact, so no scene
is excluded for lack of a route. Each cohort binds the hashes of Section 6. Its
census, the eight-per-split native replay, regression-target completion and
training admission are separate gates.
