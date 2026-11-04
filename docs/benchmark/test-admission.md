---
title: Test admission gate
icon: material/gate
---

# Test admission gate `passage-test-admission/v1`

**Table 1 — Document revision.**

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-11-03 | 0.1 — gate, record and evidence schemas, guard wiring, draft sealing record template; not independently reviewed |
| Geoffrey Delhomme | 2026-11-03 | 0.2 — after review: rehearsal aggregates bound to procedure, plan, limits and every seed; clean harness tree; committed `final-candidates.json` and hard-coded decoder digest; sealed dataset revision, root, cell and population; required bundle files; edge latency claims in the report |
| Geoffrey Delhomme | 2026-11-03 | 0.3 — G2 records campaign-session rehearsal evidence; G4 seals roster S, its block rule, unplaced cap and preparation scope; `batching` seals the dynamic admission window |
| Geoffrey Delhomme | 2026-11-04 | 0.4 — direct persistent native admission, separate replica identities and a strict Rule 8 preparation profile |

## 1. Executive summary

The 256 test pass may open the test split only after the protocol is sealed, the native chain is qualified on
validation, every edge target is integrated under a real lease, the population and the references are sealed, the
warm-up and latency rules are tested, and the candidate identities are verified. Before this gate existed, those
conditions lived in separate notes, and the harness refused the test split only through scattered path checks.

This page describes the gate that makes the conditions checkable. A *record* lists the sealed decisions, the pinned
harness, the candidate identities and the evidence files. The verifier `passage_eval.protocol.admission` evaluates six
gates, G1 to G6, and returns `GREEN` or `RED` per gate with machine-readable reasons; it never raises for a `RED`
gate. Every tool that loads or scores a test roster accepts the test split only with an explicit `allow_test` and a
record whose six gates are `GREEN` (section 5).

Section 2 lists the gates, section 3 the record and section 4 the evidence files. Section 5 gives the command and the
guarded entry points, and section 6 the limits and open decisions. Section 7 gives the unsealed draft of the protocol
256 sealing record, in which every author decision is a required field.

### 1.A. Descriptive native 2026 profile

An optional `protocol.profile: "descriptive_native_2026"` selects the narrowly scoped native measurement campaign.
Omitting it retains the existing strict checks; `"strict"` is equivalent. Unknown profiles are refused. This profile
does not qualify a reproducible entry under manuscript Rule 8, authenticate an operator, or admit a `verified`
leaderboard row. Its admission report states `confirmatory_allowed: false` and `reproducible_entry: false` in
`claim_scope`.

The scope is the public `thalesgroup/passage` revision `c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae`, `256x256`,
`elevation` with weight `62.4`, obstacle-aware tasks, exactly Seg and Reg, training seed `[2026]`, and declared targets
among Thor, Orin and Tiger. Roster S has 1,024 tasks under the existing SHA-256 selection rule and salt
`passage-public-test-v1/r0256/roster-v1/`. `population` describes those selected tasks; the distinct
`source_roster.selection.population` retains the full split's identifier inventory (40,000 in this pinned release).
Neither the full split nor a host candidate evaluation is authorised by this profile. Independent analytic references
use 65,536 expansions. A latency row still requires at least 100 warm timed requests.

The same six gates, clean committed harness, real validation rehearsal, lease evidence, reference inventory,
input identities, disjoint validation warm-up and approval checks remain required, with these explicit differences:

- `confirmatory_analysis` is sealed as `{"mode": "descriptive_only"}`. The statistics tool reports primary point
  estimates and session-based latency summaries, with no geographic block interval, p-value, Holm decision or
  geographic sensitivity interval. The campaign aggregator carries `qualification_profile` into the aggregate
  content covered by its digest. The statistics tool requires each marked aggregate's matching protocol record before
  calculation and records its profile and content digest; omitting the record is refused. Existing unmarked strict
  and validation aggregates retain their previous behavior. Deliberately removing the marker and rewriting an
  aggregate and its digests is outside this structural verifier's authenticity guarantee. Additional seeds cannot
  upgrade this profile to confirmatory inference.
- `evidence.references.historical_roster` is a file reference to the original roster v1, staged as preparation
  evidence under the authorised preparation process. G4 validates its canonical roster pin
  `0c1619faee0d743adf9c486d49e11f33c8a0d2517558de39a026cd3d5fec713c`, then compares the dataset, root receipt, full
  identifier inventory and ordered selected input identities with the new roster v2. This pin is not the JSON file
  hash or the selected-ID hash. Different v1/v2 artifact hashes are expected; missing or inconsistent history leaves
  G4 RED. The verifier never locates or opens the dataset to perform this comparison.
- Each candidate-target artifact block additionally requires `native_environment`, a pinned file reference to an
  operator attestation with schema `passage-native-environment/v1`. It binds `target`, `candidate`, `harness`,
  `checkpoint_sha256`, `bundle_sha256`, `engine_sha256`, `predictor`, `decoder`, `decoder_tree_sha256` and
  `final_decoders_sha256` to the record. It names `operator`, `observed_at`, a `stack` containing `python`, `platform`,
  `accelerator`, `driver`, `runtime`, `precision`, `power_mode` and `determinism`, and the applied `isolation`
  (`level: process` or `bwrap`, with a boolean `security_boundary`). Process separation cannot claim a security
  boundary. The record's `container_digest` must be null: no container identity is invented for a native run.
  These structural checks do not observe accelerator placement or prove the actual loaded files by themselves.
- Exceeding the unplaced-footprint cap stops block inference while retaining descriptive task and latency reports.
  It does not exclude or replace tasks. The strict profile keeps its existing cap gate.

The standalone `run_supervised` entry point refuses this grant before opening even the roster: it has no attested
edge-target identity. Only an admitted edge campaign may execute it; the campaign planner checks its seed, Seg/Reg
procedures, target, budgets and artifact bindings. The on-target split-worker runner can still consume authorised
input-only packages. A preparation grant continues to authorise only preparation readers, never a candidate run.
This profile defines admission checks; it does not implement or qualify the native edge runner's integration.

### 1.B. Persistent, non-supervised native timing

Within `descriptive_native_2026`, `timing_profile: native_raw_path_persistent_v1` selects a separate non-supervised
measurement. Its schema is `passage-complete-request-native-v1`, its boundary is
`passage-timing-boundary/v6-native-raw-arrays-to-path`, and the scorer uses `native_wall_ms`: raw arrays already in
target memory through tensor construction, native inference, synchronisation, decoding and emitted path. IPC is
not required for this mode. Validation and physical recosting happen separately on the host. Their measured time is
`validation_recost_ms`; `total_wall_ms` is only their sum with `native_wall_ms`, not target latency. Budget and failure
accounting use `native_wall_ms`, so host recosting cannot turn an on-time native path into a timeout.

One persistent process serves each candidate on each target. The sealed `paired_order.granularity` is
`per_candidate`, `max_restarts` is zero, and `r7_policy` is
`{"mode": "abort_incomplete_session_no_automatic_replay"}`. An incomplete session is retained for diagnosis, never
automatically replayed or scored as a complete pass. An explicitly documented continuation may use a fresh process
and its own initial warm-up, retaining the accepted prefix and executing only uncompleted tasks. Before the next candidate starts, both the previous process and
its GPU context must be released. Input fragments and journal checkpoints do not create new worker sessions; no
group of 32 scenes is required.

The sealed `warmup` decision retains its salt and dataset-root pin, with `k: 1` and `repeats: 16`: one validation
scene disjoint from the measured inputs, repeated 16 times initially, never between fragments. The warm-up roster
therefore contains one scene. A native artifact identifies its profile under `procedure.execution.timing_profile`
and has `native_execution` with an opaque `session_id`, `n_processes: 1`, `n_restarts: 0`, and `warmup` containing
`k: 1`, `repeats: 16`, `performed: true`, `disjoint_by_content: true`, and `input_roster_sha256`. That last digest is
the input-only projection; canonical host rosters, Parquet hashes, references and labels stay off the target.

For this mode, G2 uses the per-target `evidence.edge` receipts instead of a supervised host rehearsal. Each has schema
`passage-native-persistent-validation/v1`, its `target`, `split: validation`, `harness`, `plan_sha256` and exactly
Seg/Reg under `procedures`. Each procedure names its pinned `aggregate`, checkpoint and engine hashes, records
`completed: true`, `accelerator_used: true`, and its `native_execution`. The aggregate must preserve the canonical
scorer, population, budget and timing. The functional rehearsal needs at least one completed warm observation per
candidate and target; the planned validation pass uses 128 scenes. The 100-observation minimum remains a requirement
of a final latency row, not an artificial prerequisite of preparation. No artificial re-warm,
supervised batch kill or host candidate pass is required. G3 checks each procedure's lease (resource, holder, activity,
job and acquire/release times) and `process` (start, finish, drain times, `process_exited: true` and
`gpu_context_released: true`); the previous drain precedes the next candidate start. A hardware inventory, SSH-key
inventory and additional transport-failure campaigns are not required by this native branch. Strict gates retain
their existing checks.

When the lease service retains event order but not exact historical lease timestamps, an alternative G3 evidence
path uses pinned `cavops_observation`, `image_start`, `image_finish`, `drain` and `journal` references per procedure.
It checks the same completed CAVIAR job and exclusive resource, the observed process and journal identity, runtime
release and an empty stopped scope. It binds the observed image, configuration, ordered raw-input index and candidate
artifacts to the target receipt and the G2 run manifest. The two procedures must use the same event ledger, and the first job's terminal
event must precede the second job's start event. This relies on the CAVIAR supervisor's transaction: its terminal
event and lease deletion are committed together. Event snapshots are evidence supplied by the operator, not a
cryptographic attestation; process exit closes session descriptors but does not independently inspect driver contexts.
The original timestamped receipt path remains supported.

Each procedure additionally pins `roster`, `warmup_roster` and `run_manifest` on the host. G2 compares the actual input
identities, checks disjointness by identifier and content, verifies the warm-up projection digest, and binds the
session and executed checkpoint/engine/decoder digests to both the emitted manifest and aggregate. The manifest's
records/path hashes are pinned too. G5 must retain the same verified warm-up input. These are structural links;
unattested declarations alone do not demonstrate GPU placement.

For this persistent native mode, G5 checks the fixed validation warm-up directly against roster S and the admitted
measured roster by sample ID and input-content identity. Its selection record keeps its actual original selection provenance;
it need not claim that the warm-up was selected against the later roster. A collision stops admission. The supervised
profile retains its existing evaluated-roster derivation check.

Native latency statistics report descriptive p50/p95/p99, without a session-bootstrap or paired latency interval.
Multiple processes after a documented interruption remain descriptively aggregable: their request IDs must cover the
scored tasks exactly once, every process records its initial warm-up, and `native_resumes` binds consecutive session
IDs, a reason, the prior accepted-session prefix hash and its journal digest. Input fragments are not process resumes.
The legacy profile remains restricted to 2026. A separate `descriptive_native_single_seed` record may seal exactly one
of 1234, 2026, 3407, still on S only; 2026 remains the initial pass. Later seeds require the pinned
`evidence.replica_catalog` file reference. The separate `replica-candidates-256.json` catalogue binds their four
accepted checkpoint identities to the file digest of the unchanged 2026 manifest and its decoder digest. G6 checks
the selected seed's checkpoint, bundle lineage and target environment against that catalogue. A validation plan may
name the same file with an absolute `replica_catalog: {path, sha256}`; an admitted test plan obtains its identities
from its sealed record. Input packages contain only the selected checkpoint identity and seed, without the host
catalogue. These pins identify weights; they do not establish target execution or qualification. The 2026 manifest
and decoder remain frozen; its checkpoint cannot be relabelled as another seed. This descriptive mode does not establish
the strict reproducible entry required by Rule 8. A future 40,000-scene objective is not authorisation to prepare, read
or evaluate the rest of the test split.

### 1.C. Direct measurement toward a Rule 8 entry

`protocol.profile: "rule8_native_persistent"` admits preparation of the same direct measurement under the strict
reproducibility requirements. The original `strict` profile keeps its supervised checks. The new profile retains the
scope of S, its historical snapshot and ordered input identities, Seg/Reg, edge targets, one planned training seed
per record, initial warm-up, sequential processes and full release between candidates. It requires the native
timing profile and strict comparison of limits. No IPC measurement or groups of 32 are required by this branch.

The three seed records seal the same `confirmatory_analysis` for `[1234, 2026, 3407]`, 10,000 shared paired draws
and 95% confidence intervals (`alpha: 0.05`) before the first test pass, as specified in Statistical Discipline.
The complete separate replica catalogue is pinned even for the initial 2026 record, fixing all four later-seed
checkpoint identities before the first pass. This is a choice of the expressly approved S campaign, whose replicas
already exist; it is not a general requirement attributed to the paper. Other campaigns retain their existing
strict profile. Each candidate-target block requires an actual `container_digest`
and a pinned `container_reconstruction` description. An existing `native_environment` may be retained and checked;
it is optional here and does not replace image evidence. No additional version attestation is required: target
configuration may reuse the pinned compiled manifest and the actual runtime observations. The reconstruction description has schema
`passage-container-reconstruction/v1`, `target`, `container_digest`, the exact `harness`, `source_repository`, a
`build_command` argument list, a `recipe` file reference and a nonempty `dependencies` list of file references.
Dependencies may record the base image, locks, wheels or configuration used by the actual recipe. G6 verifies these
files and pins them for later grant checks. `native_execution.container_digest` must equal the sealed image digest;
the existing session checks bind it to the validation receipt, emitted run manifest and aggregate.
The host must obtain this digest from an observation of the actual launcher and image; the target must not claim
to have observed its own container identity merely because the field was supplied to it.

These checks describe and link evidence. They cannot authenticate a producer, observe accelerator placement,
establish that a repository is published, or demonstrate an image rebuild by themselves. A diagnostic rootfs hash
or local image configuration ID must not be labelled as a container manifest digest. Actual published source,
rebuildable images and observed execution still have to be produced. Missing evidence leaves admission RED.
Before three-seed inference, the statistics tool also requires the same catalogue file digest in all three records.
This establishes a constant commitment; hash equality alone cannot establish when it was sealed. The approved
records and their actual approval and execution chronology must retain that separate evidence.

An all-GREEN admission permits only the sealed passes. It does not mark a leaderboard entry reproducible or verified.
Rule 8 additionally requires the canonical procedure, calibration decisions and their pinning hash, a complete
sidecar for each reported row, and the seed-replicated paired-bootstrap confidence intervals. The statistics tool
accepts marked aggregates only with their matching sealed records. With three seeds it uses the existing paired
quality analysis and the predeclared block criterion. This criterion is a local inferential condition, not a new
requirement attributed to the manuscript. If blocks fail it, intervals remain descriptive and no confirmatory
p-value, Holm decision or superiority claim follows. Native latency remains a per-session descriptive quantile
summary: one persistent process does not create independent sessions for a bootstrap. No interval is invented.

## 2. The six gates

**Table 2 — Gates of `passage-test-admission/v1`.** Every gate must be `GREEN` before the test split is opened.

| Gate | Title | What is checked |
|---|---|---|
| G1 | protocol sealed and approved | record status `sealed`; the sealed protocol document matches its SHA-256 and carries no `proposed_not_sealed` marker; every decision of section 7 is present, valid and `sealed`, with no `draft`, `proposed_not_sealed`, `todo` or `placeholder` value; the authorised harness commit is the record's commit; an approval names the approver, the time (not in the future) and the SHA-256 of the approved record |
| G2 | native chain validated on validation | the running harness is the record's commit, with no uncommitted change under `src`, `scripts`, `docs/benchmark/final-candidates.json`, `pyproject.toml` or `uv.lock`, and the record's source hash; a `passage-native-chain-rehearsal/v1` receipt of a host rehearsal on the validation split, made with that harness, naming its rehearsal plan, with the frozen candidates, the sealed budget and timing profile, a performed warm-up, an exercised re-warm, a deliberately killed batch relaunched under the planned R7 rule with no killed attempt scored, and an exercised resume without any unplanned relaunch; one distinct campaign aggregate per sealed procedure, of that procedure, the host, the validation split, that plan and harness, the sealed budget, timing profile and seeds, with consistent population counts and one completion digest per batch or R7 unit; for every sealed seed, an aggregation equal to the canonical scorer and a latency block with at least the sealed minimum of warm requests; no marker of synthetic, simulated or unsealed content in the receipt or the aggregates |
| G3 | edge leases integrated per target | for every edge target of the sealed targets (`thor`, `orin`, `tiger`): a `passage-edge-lease-evidence/v1` file with a target identity attestation, the lease activity and job identifiers, a continuous lease verification log covering the run and the drain, a drain confirmed before release, and transport-loss and cancellation test receipts; identities and leases distinct between targets. An edge target outside the sealed targets is not required; the report states, per edge target, whether an edge latency claim is admitted |
| G4 | references and roster sealed | the source roster (roster S, `passage-roster-v2`, every footprint required or the sealed `unplaced` block declared, complete block table bound to the installed block rule) of the sealed dataset revision (repository, commit, configuration, resolution), dataset root, evaluation cell and population (`n_rows` and `ids_sha256`), with the sealed block rule and degrees and the sealed treatment of missing footprints (`block_analysis`); the reference admission policy, inventory, admissible roster and chain, re-derived by `admission_chain` and equal to the recorded chain; the admissible roster of the same revision, root and cell; policy, cohort, search limit, category table and exclusion caps equal to the sealed decisions. The block adequacy and unplaced share of the admissible roster are reported (`admitted.block_adequacy`); a cap breach halts block inference but does not turn the descriptive campaign into a failure |
| G5 | warm-up and latency tested | a validation `passage-roster-v2` warm-up roster built against the admissible roster of G4, disjoint by input content, with the sealed salt and size, on the dataset root, revision and cell of the admissible roster, and its selection record; the sealed censoring rule is the installed `passage-censored-latency/v1`; the rehearsal latency blocks of G2 meet the sealed minimum for every sealed seed |
| G6 | artifacts and candidate identities verified | `docs/benchmark/final-candidates.json` equals the file committed at the record's harness commit; its checkpoint SHA-256 values equal the record's; the installed `final_decoders.py` and the file's decoder pin both equal the frozen SHA-256 held by the verifier (`2a25ad31…cbafc4`); frozen decoder entries; the decoder tree pin equals the tree digest of the installed decoder package; for every target a bundle file whose digest is the declared bundle digest and whose lineage names the checkpoint, and an engine and a container digest for every edge target |

> **Why G3 is `RED` today**
>
> No real `cavops`-leased remote run with a qualified transport exists. Absent evidence is `RED`. Evidence that
> declares itself synthetic, loopback, simulated, mocked, a stub, a template or unqualified (`qualified: false`,
> `target_evidence: none`) is `RED` too. The campaign runner refuses every edge target with
> `EdgeTransportNotQualified`, even if G3 were `GREEN`, because no remote transport is implemented.

## 3. The record

A record is one JSON object with exactly the keys of Table 3; an unknown or missing key turns every gate `RED`
(`record_structure_invalid`). Paths are relative to the evidence directory (default: the record's directory);
absolute paths, `..` and symbolic links are refused. A file reference is exactly `{"path", "sha256"}`.

**Table 3 — Keys of a `passage-test-admission/v1` record.**

| Key | Content |
|---|---|
| `schema`, `status`, `split`, `record_id` | `passage-test-admission/v1`; `sealed` or `draft`; `test`; a name |
| `protocol` | `version`, `document` (file reference of the sealed protocol) and `decisions` (section 7), each `{"value", "status", "note"}` with status `sealed`, `proposed_not_sealed` or `required` |
| `harness` | `commit` (40 hexadecimal digits) and `source_sha256` (`passage_eval.cli._source_hash`); G2 requires a clean checkout of that commit |
| `candidates` | `seg` and `reg`: `checkpoint_sha256`, `predictor`, `decoder`, `decoder_tree_sha256`, `final_decoders_sha256`, and per target `bundle` (file reference of the bundle manifest, required by G6), `bundle_sha256`, `engine_sha256`, `container_digest` |
| `evidence` | `native_chain_rehearsal`; `edge` (`thor`, `orin`, `tiger`); `references` (`source_roster`, `policy`, `inventory`, `admissible_roster`, `chain`); `warmup` (`roster`, `selection`) |
| `approval` | `approver`, `approved_at` (ISO 8601 with a time zone) and `approved_record_sha256` |
| `notes` | optional free text |

The approval names `record_core_sha256(record)`: the SHA-256 of the canonical JSON of the record without its
`approval` block. Any change to the record after the approval, including a new evidence digest, makes G1 `RED` until
the approver approves the new record.

## 4. Evidence files

**Table 4 — Evidence schemas.**

| Schema | Produced by | Main fields |
|---|---|---|
| `passage-native-chain-rehearsal/v1` | the operator, from a validation campaign | `split`, `target`, `plan_sha256` (the rehearsal plan), `harness`, `candidates`, `budget`, `timing_profile`, `warmup {performed, roster_sha256, k}`, `rewarm {exercised, n_rewarm_events}`, `deliberate_kill {exercised, n_deliberate_kills, n_planned_relaunches, killed_attempts_scored}`, `resume {exercised, n_batches_resumed, n_unplanned_relaunches}`, `procedures {name: {aggregate}}`; built by `python -I -m passage_eval.campaign rehearsal-receipt` from the real outputs of a qualifying rehearsal (`docs/benchmark/campaign.md`, section 7) |
| `passage-campaign-aggregate/v1` | `python -I -m passage_eval.campaign aggregate` | `procedure`, `plan_sha256`, `campaign_id`, `split`, `target`, `seeds`, `harness`, `admissible_roster_sha256`, `reference_admission_chain_sha256`, `budget`, `timing_profile`, population counts, `receipts` (one completion digest per batch or R7 unit), `cold_start`, `sessions`, `restarts`, `relaunches`, `imputed_failures`, parent blocks, `canonical_equivalence` over every seed, and `per_seed` (six metrics, failures, censored latency block, per-batch equivalence and the loaded checkpoint or bundle digests of each seed) |
| bundle manifest | the export of the candidate | `lineage.checkpoint_sha256`, read by G6; the file digest is the declared `bundle_sha256` |
| `passage-edge-lease-evidence/v1` | a real leased remote run (no such run exists) | `target`, `harness`, `target_identity {hostname, machine_id_sha256, ssh_host_key_sha256, agent_sha256}`, `lease {resource, activity_id job:<16 hex>, job_id, holder_host edge-<target>, effect_key, acquired_at, released_at}`, `run`, `lease_verification_log {path, sha256, period_s, n_checks}`, `drain {cgroup_empty, accelerator_idle, before_release, confirmed_at}`, `transport_loss_test`, `cancellation_test` |
| `passage-edge-lease-check/v1` | one line per lease check of the verification log | `checked_at`, `activity_id`, `resource`, `holder_host`, `confirmed` |
| `passage-edge-fault-test/v1` | a transport-loss or cancellation test on the target | `target`, `kind`, `outcome`, `drain_confirmed`, `lease_retained_until_drain`, `activity_id` |
| reference admission files | `python -I -m passage_eval.protocol.reference_admission` | policy, inventory, admissible roster and chain (supervised-execution guide, section 3.F) |
| warm-up files | `scripts/build_warmup_roster.py` | warm-up roster and its `passage-warmup-selection/v1` record |

The lease log must confirm the same activity at intervals of at most twice its period, from before the run starts to
after the drain. The lease times must be ordered: acquire, start, finish, drain, release.

Marker scans. The rehearsal receipt, the aggregates and the edge evidence are scanned for the markers of synthetic,
loopback, simulated, mocked, stub, template or unqualified content; the receipt and the aggregates also for unsealed
markers (`draft`, `proposed_not_sealed`, `todo`, `placeholder`, `unsealed`). One label is exempt: the `status` that
`metrics.latency` writes in a latency block of its own rule (`passage-censored-latency/v1`, `proposed_not_sealed`
until the code seals the rule). G1 binds the author's sealing of that rule and G5 binds it to the installed rule.

## 5. Command and enforcement

### 5.A. `passage-eval admission-check`

```bash
passage-eval admission-check RECORD.json [--evidence-dir DIR] [--json] [--final-candidates FILE]
passage-eval admission-check --template > protocol-256-sealing-record.json
```

The command prints the gate table, or the machine-readable report (`passage-test-admission-check/v1`) with `--json`.
It exits with 0 when all six gates are `GREEN` and with 1 otherwise: a `RED` gate is a verdict, not an error.
`--template` prints the draft record of section 7. Both outputs state, per edge target, whether an edge latency claim
is admitted (`edge_latency_claims`): only a sealed target of an all-`GREEN` record is; the others are listed as not
admitted, so that a record sealed for the host alone, or for part of the edge targets, cannot be read as edge
evidence for the rest.

### 5.B. Guarded entry points

`passage_eval.protocol.guards` accepts the test split only with `allow_test=True` and an `AdmissionGrant`. A grant is
created only by `admission.grant_test_admission` (or by `guards.test_split_allowed` for a command) from an all-`GREEN`
report. It re-hashes its record, every evidence file it pinned, `final-candidates.json` and the installed
`final_decoders.py` before each use, so an edit after the check is refused. Commands take `--allow-test-split`, `--test-admission-record FILE` and `--test-admission-evidence-dir DIR`;
the flag without a record, or a record without the flag, is refused.

**Table 5 — Entry points that refuse the test split before any file of the split is opened.**

| Entry point | Additional binding |
|---|---|
| `scripts/build_public_test_roster.py` (`build_roster` and the command) | preparation reader `build_roster` |
| `protocol.block_census` (`block_census` and the command) | preparation reader `block_census`; reads no dataset file |
| reference admission (`admission_policy`, `build_reference_inventory`, the command) | preparation reader `reference_inventory` |
| `build_warmup_roster` and `scripts/build_warmup_roster.py` (an evaluated roster of the test split) | preparation reader `warmup_roster`; the warm-up inputs themselves are refused on test even with a grant |
| `python -m passage_eval.supervision run` and `run_supervised` (evaluated roster and spec split) | the roster must be the sealed admissible roster or a child derived from it; a spec run on test is never admitted; every re-qualified reference must carry the inventory's `reference_id` |
| the supervisor's warm-up sources (`WarmupInput`, `--allow-test-warmup`) | none (the protocol takes warm-up inputs from validation) |
| `score-roster` (`score_roster_run` and the command) | the roster must be admitted as above, with the inventory's `reference_id` for every task |
| `protocol.warmup` selection | the test split is refused even with a grant |
| `passage_eval.campaign` plan, partition, runner and aggregator | the plan must repeat the record's sealed decisions, rosters and candidate identities (campaign tooling guide, section 3) |

`stage-score`, `passage-eval score`, `passage-eval bench`, `compare` and `campaign statistics` (which reads campaign
aggregates only) are not guarded. The latter is an optional post-campaign analysis tool, outside the current
single-seed qualification. The tests of `tests/passage_eval/test_test_split_canary.py` place a mode-000 canary
`test` directory next to a validation directory and record every `open`, `stat`, `lstat`, `scandir` and `listdir`
below it: each refusal leaves the record empty, including a fixture-sized roster rejected by the block census before
it can exceed the sealed roster-S scope.

### 5.C. Phased preparation

G4 needs the test source roster, its block census, its reference inventory and the admissible roster, and G5 the
warm-up roster built against that admissible roster. Only reading the test split produces them, so a record cannot be
all `GREEN` before they exist. `admission.grant_test_preparation(record, approval, evidence_dir)` (or a preparation
reader's command with `--allow-test-split`, the record and `--test-preparation-approval FILE`) returns a
`PreparationGrant` under three conditions:

1. The record is checked by the six gates and its only `RED` reasons are the missing G4 and G5 artefacts
   (`PREPARATION_ARTEFACT_REASONS`: the five reference files and the two warm-up files) and, while the roster builder
   has not inventoried it yet, the `population` decision left `required` with a null value. Every other decision must
   be sealed and approved, G2, G3 and G6 `GREEN`; an artefact that is referenced but broken refuses the preparation.
2. A separate `passage-test-preparation-approval/v1` file of the author (Table 7) names the SHA-256 of the record file,
   the readers it allows, the approver, a statement and a window that contains the present time.
3. The window is at most `PREPARATION_MAX_WINDOW` (72 hours, a harness bound) long.

**Table 7 — Keys of a `passage-test-preparation-approval/v1` file.**

| Key | Content |
|---|---|
| `schema`, `scope` | `passage-test-preparation-approval/v1`; `preparation` |
| `readers` | nonempty subset of `build_roster`, `block_census`, `reference_inventory`, `warmup_roster` |
| `admission_record_sha256` | SHA-256 of the record file's bytes at approval |
| `approver`, `approved_at`, `expires_at`, `statement` | the author, the window (ISO 8601 with a time zone) and the author's statement |

The guards accept a preparation grant only for a call that names a reader the approval lists. A run, a warm-up source,
the roster scorer and the campaign tools name no reader, so they refuse it with "authorises only …"; a preparation
grant admits no roster for a run. Before each use the grant re-hashes the record, the evidence and frozen files of the
check and the approval file, and refuses after `expires_at`. The usual order is: seal and approve the record without
the G4 and G5 files; approve the preparation; build the source rosters and their block census; seal the `population`
and re-approve; build the reference inventory and the admissible roster; reference them, re-approve and approve the
preparation again; build the warm-up roster; reference it and approve the complete record, which must then be all
`GREEN` for any run.

An existing authorization may instead be recorded without claiming that the author approved later file hashes.
The record's `approval` then contains `authorization` and `operator_receipt`. The authorization records the actual
statement and its source, `approver` and `approved_at` when known (otherwise null), and `scope_sha256` computed with
`authorization_scope_sha256(record)`. This digest is an operator's translation of the authorized scope, not a
signature or proof that the author saw a digest. The scope covers the record except its approval, notes, population
decision and generated rehearsal, edge, reference and warm-up evidence. The pinned replica catalogue remains in the
scope. The operator receipt has `operator`, `created_at`,
`record_core_sha256` and `authorization_sha256` (the canonical JSON digest of the authorization). Its time is the
receipt creation time, not an invented authorization time. A changed material scope refuses G1; after adding generated
evidence, the operator updates the record-core binding. The verifier checks structure and hashes, not authorship or
whether the source faithfully represents the actual authorization.

For phased preparation under this alternative, `--test-preparation-approval` can name a
`passage-test-preparation-operator-receipt/v1` JSON with `scope: "preparation"`, `readers`, the current record's
file-byte digest `admission_record_sha256`, the canonical `authorization_sha256`, `operator`, `created_at` and
`expires_at`. Its validity window is technical and at most 72 hours. It is not a new human approval. The existing
author approval format remains available. Both formats retain the same gate, scope and unchanged-file checks.

### 5.D. Operator connection configuration

Direct edge relays require the operator to configure the SSH alias and machine ID for each target before a queue is dispatched. Set `PASSAGE_EDGE_SSH_HOST_<TARGET>` to the intended SSH host or alias and `PASSAGE_EDGE_MACHINE_ID_<TARGET>` to its 32-character lowercase hexadecimal machine ID, where `<TARGET>` is `THOR`, `ORIN` or `TIGER`. These values are deployment configuration, not benchmark evidence. The remote machine ID is checked before execution, and the SSH host key remains pinned by the existing specification. Missing or malformed settings fail closed. The job launcher must pass the settings to the relay process; keep account-specific values outside the source tree.

The historical `leased-run` driver uses the current working directory for its lease client. Operators can supply a different `caviar_root` when embedding the driver, and can select the executable with `PASSAGE_EDGE_LEASE_COMMAND` or the driver's `lease_command` argument. The existing lease protocol and direct-session specification keys remain unchanged. Configuring a client or an SSH host does not establish admission, a held lease or device identity.

The legacy pre-G3 validation relay also requires `PASSAGE_VALIDATION_ROSTER` and `PASSAGE_VALIDATION_DATASET_ROOT`, both absolute paths supplied by the operator. The roster is not included in the public source projection. Only locations are configurable: the harness keeps the existing roster-file, roster and dataset-receipt SHA-256 pins and checks them before reading validation scenes. Missing settings leave this relay unconfigured. Synthetic test fixtures exercise the checks without publishing the historical roster or reading the test split.

Historical records whose host-verifier binding was created against the earlier verifier must be replayed with the corresponding retained private checkout at commit `7e9808e` and their original runtime binding. This portable checkout has different source bytes; configuring paths does not relabel those runs or make the two verifier versions interchangeable.

## 6. Limits and open decisions

- **No authentication.** Every check is structural, numerical or a hash comparison. A hash proves byte
  correspondence, not authorship. Complete, consistent evidence without the explicit markers is accepted; the approval
  binds its digests, and the approver remains responsible for its authenticity. The verifier contacts no target, lease
  registry or network service.
- **Preparation order.** The phased preparation of section 5.C is implemented as a proposal; whether the protocol
  admits it, and its window, are the author's decisions. It tolerates one G1 reason (the `population` decision still
  `required`), because the roster builder inventories that population; every other sealed decision must precede the
  preparation. The reference inventory reads the labels of the test split under this grant.
- **Block adequacy is reported, not gated.** A failed adequacy criterion makes the block analysis descriptive only;
  an unplaced share above the sealed cap halts that block analysis, while the descriptive task and latency reports
  remain available.
- **Host-only and partial edge campaigns.** G3 checks the sealed edge targets only; a record whose sealed targets
  are `["host"]` has a vacuous G3, and one sealed for `["host", "thor"]` needs no Orin or Tiger evidence. Whether the
  protocol requires every edge target is the author's `targets` decision; the report lists the edge targets for which
  no latency claim is admitted.
- **What G2 does not prove.** The rehearsal receipt and its aggregates are checked for consistency with the record,
  with each other and with the sealed decisions; the batch completion receipts they list are not in the evidence
  directory and are not re-verified, and the verifier does not rerun the rehearsal. The rehearsal is on validation:
  its admissible roster and chain digests are checked for form and consistency between procedures, not against G4.
- **Clean checkout.** G2 runs `git status` and G6 `git show` on the checkout of the installed harness; a harness
  installed from a wheel, or a checkout with uncommitted changes under the harness paths, is `RED`.
- **One frozen checkpoint per candidate.** `final-candidates.json` freezes one checkpoint per candidate, trained with
  seed 2026. A campaign plan with Seg or Reg runs only that seed; a confirmatory multi-seed campaign needs a frozen
  checkpoint per seed, which the protocol author has not frozen.
- **Not decided here.** Every value of section 7 belongs to the protocol author; the template seals nothing.

## 7. Draft sealing record for protocol 256

The record below is the output of `passage-eval admission-check --template`. It is an unsealed draft: status
`draft`, no protocol document, no evidence, no approval, and every decision `required` with a null value. Table 6 maps
each decision to the open question it closes (numbers of the harness gap analysis, track decisions D1 to D8 and
transport gates P1 and P2).

**Table 6 — Author decisions of the sealing record.**

| Decision | What the author fixes |
|---|---|
| `dataset_revision` | {repo, commit (40 hex), config, resolution} of the retained published dataset revision (protocol section 1) |
| `dataset_root_sha256` | SHA-256 of the pinned dataset root receipt (dataset_root_identity) of that revision |
| `evaluation_cell` | {cost_model, cost_weight, task_variant} of the evaluated cell (protocol section 1) |
| `population` | {n_rows, ids_sha256} of selected source roster S, inventoried before the run; the full split inventory remains in `source_roster.selection.population` |
| `budget` | per-request budget {wall_time_ms, max_expansions, max_queue_operations, max_live_states} (decision 1, D6) |
| `timing_profile` | timing profile of the protocol: supervised_service_v1 or supervised_service_rewarm_v1 (D1) |
| `timing_boundary_id` | timing boundary identifier recorded by that profile (D1) |
| `deadline_grace_ms` | grace after the wall deadline before preemption (decision 1, D6) |
| `hard_timeout_ms` | hard timeout of a request without a wall budget (D4) |
| `startup_timeout_s` | worker start timeout (D4) |
| `max_restarts` | restart cap per seed (D2, D4); at least the sealed batch size (R7) |
| `worker_rlimits` | worker resource limits {nofile, fsize_mb, as_mb, cpu_s} (D4) |
| `reference_policy` | released_labels or independent_analytic_reference (decision 2) |
| `reference_cohort_id` | named cohort of the independent policy; null for released_labels (decision 2) |
| `reference_max_expansions` | exact reference search limit, at least N*N; 65536 and 65537 give other ids (decision 2) |
| `reference_category_table_sha256` | SHA-256 of the sealed exclusion/halt category table (decisions 3, 10) |
| `reference_exclusion_caps` | {max_total_fraction, max_block_fraction, min_block_exclusions} of the reference inventory (judge R1) |
| `block_analysis` | {rule, block_degrees, missing_footprint, min_blocks, max_block_share, max_unplaced_fraction}: block rule and adequacy criterion sealed before the label-free census (judge R5, tightening 1) |
| `censoring_rule` | latency censoring rule {id} used for percentiles (decision 4, D3) |
| `min_warm_requests` | minimum warm requests of a latency row (manuscript: 100) |
| `cold_start_policy` | what is reported of the cold start (decision 5) |
| `restart_and_failed_warmup_policy` | treatment of a restart and of a failed warm-up (decision 6, D2, D8) |
| `paired_order` | {granularity, procedures, schedule_seed} of the paired order (decision 7) |
| `batching` | {batch_size (1..1024), lease_or_timeout_ms, fixed_overhead_ms, per_seed_overhead_ms, per_request_overhead_ms, persistence_ms, observed_request_ms {procedure: ms}, safety_margin (>= 3)}: measured overheads and window of dynamic request admission (decision 8, D4, R7) |
| `killed_batch_policy` | a killed batch: imputed failures or ambiguous case (decision 8) |
| `candidate_start_failure_policy` | a candidate that cannot start: candidate failure or infrastructure (decision 9) |
| `missing_footprint_policy` | a task without footprint: halt of block analysis or declared exclusion (decision 10) |
| `warmup` | {salt, k, dataset_root_sha256 (the sealed dataset root)} of the warm-up roster (decision 11) |
| `infrastructure_relaunch_policy` | traced relaunch policy of a batch failed by infrastructure (decision 12) |
| `harness_commit_authorization` | {commit} of the harness authorised and committed on GitLab (decision 13) |
| `targets` | targets of the campaign among host, thor, orin, tiger |
| `edge_timing_boundary` | timing boundary of an edge row: network included or on-target timing (P1) |
| `infrastructure_failure_policy` | transport or lease loss: batch invalidated, never a candidate penalty (P2) |
| `seeds` | training seeds evaluated (protocol: 2026 preliminary; 1234, 2026, 3407 confirmatory) |
| `strict_comparison` | whether protocol comparisons require identical recorded limits (D5) |

**Schema 1 — Draft `passage-test-admission/v1` sealing record (all decisions required, nothing sealed).**

```json
{
  "approval": null,
  "candidates": {
    "reg": {
      "checkpoint_sha256": null,
      "decoder": null,
      "decoder_tree_sha256": null,
      "final_decoders_sha256": null,
      "predictor": null,
      "targets": {
        "host": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "orin": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "thor": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "tiger": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        }
      }
    },
    "seg": {
      "checkpoint_sha256": null,
      "decoder": null,
      "decoder_tree_sha256": null,
      "final_decoders_sha256": null,
      "predictor": null,
      "targets": {
        "host": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "orin": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "thor": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        },
        "tiger": {
          "bundle": null,
          "bundle_sha256": null,
          "container_digest": null,
          "engine_sha256": null
        }
      }
    }
  },
  "evidence": {
    "edge": {
      "orin": null,
      "thor": null,
      "tiger": null
    },
    "native_chain_rehearsal": null,
    "references": {
      "admissible_roster": null,
      "chain": null,
      "inventory": null,
      "policy": null,
      "source_roster": null
    },
    "warmup": {
      "roster": null,
      "selection": null
    }
  },
  "harness": {
    "commit": null,
    "source_sha256": null
  },
  "notes": "Draft template: every decision is required and unsealed; no evidence is attached.",
  "protocol": {
    "decisions": {
      "batching": {
        "note": "{batch_size, lease_or_timeout_ms, fixed_overhead_ms, per_seed_overhead_ms, per_request_overhead_ms} (decision 8, D4)",
        "status": "required",
        "value": null
      },
      "block_analysis": {
        "note": "{rule, block_degrees, missing_footprint, min_blocks, max_block_share, max_unplaced_fraction}: block rule and adequacy criterion sealed before the label-free census (judge R5, tightening 1)",
        "status": "required",
        "value": null
      },
      "budget": {
        "note": "per-request budget {wall_time_ms, max_expansions, max_queue_operations, max_live_states} (decision 1, D6)",
        "status": "required",
        "value": null
      },
      "candidate_start_failure_policy": {
        "note": "a candidate that cannot start: candidate failure or infrastructure (decision 9)",
        "status": "required",
        "value": null
      },
      "censoring_rule": {
        "note": "latency censoring rule {id} used for percentiles (decision 4, D3)",
        "status": "required",
        "value": null
      },
      "cold_start_policy": {
        "note": "what is reported of the cold start (decision 5)",
        "status": "required",
        "value": null
      },
      "dataset_revision": {
        "note": "{repo, commit (40 hex), config, resolution} of the retained published dataset revision (protocol section 1)",
        "status": "required",
        "value": null
      },
      "dataset_root_sha256": {
        "note": "SHA-256 of the pinned dataset root receipt (dataset_root_identity) of that revision",
        "status": "required",
        "value": null
      },
      "deadline_grace_ms": {
        "note": "grace after the wall deadline before preemption (decision 1, D6)",
        "status": "required",
        "value": null
      },
      "edge_timing_boundary": {
        "note": "timing boundary of an edge row: network included or on-target timing (P1)",
        "status": "required",
        "value": null
      },
      "evaluation_cell": {
        "note": "{cost_model, cost_weight, task_variant} of the evaluated cell (protocol section 1)",
        "status": "required",
        "value": null
      },
      "hard_timeout_ms": {
        "note": "hard timeout of a request without a wall budget (D4)",
        "status": "required",
        "value": null
      },
      "harness_commit_authorization": {
        "note": "{commit} of the harness authorised and committed on GitLab (decision 13)",
        "status": "required",
        "value": null
      },
      "infrastructure_failure_policy": {
        "note": "transport or lease loss: batch invalidated, never a candidate penalty (P2)",
        "status": "required",
        "value": null
      },
      "infrastructure_relaunch_policy": {
        "note": "traced relaunch policy of a batch failed by infrastructure (decision 12)",
        "status": "required",
        "value": null
      },
      "killed_batch_policy": {
        "note": "a killed batch: imputed failures or ambiguous case (decision 8)",
        "status": "required",
        "value": null
      },
      "max_restarts": {
        "note": "restart cap per seed (D2, D4)",
        "status": "required",
        "value": null
      },
      "min_warm_requests": {
        "note": "minimum warm requests of a latency row (manuscript: 100)",
        "status": "required",
        "value": null
      },
      "missing_footprint_policy": {
        "note": "a task without footprint: halt of block analysis or declared exclusion (decision 10)",
        "status": "required",
        "value": null
      },
      "paired_order": {
        "note": "{granularity, procedures, schedule_seed} of the paired order (decision 7)",
        "status": "required",
        "value": null
      },
      "population": {
        "note": "{n_rows, ids_sha256} of the selected source roster S (not selection.population's full split inventory), inventoried before the run",
        "status": "required",
        "value": null
      },
      "reference_category_table_sha256": {
        "note": "SHA-256 of the sealed exclusion/halt category table (decisions 3, 10)",
        "status": "required",
        "value": null
      },
      "reference_cohort_id": {
        "note": "named cohort of the independent policy; null for released_labels (decision 2)",
        "status": "required",
        "value": null
      },
      "reference_exclusion_caps": {
        "note": "{max_total_fraction, max_block_fraction, min_block_exclusions} of the reference inventory (judge R1)",
        "status": "required",
        "value": null
      },
      "reference_max_expansions": {
        "note": "exact reference search limit, at least N*N; 65536 and 65537 give other ids (decision 2)",
        "status": "required",
        "value": null
      },
      "reference_policy": {
        "note": "released_labels or independent_analytic_reference (decision 2)",
        "status": "required",
        "value": null
      },
      "restart_and_failed_warmup_policy": {
        "note": "treatment of a restart and of a failed warm-up (decision 6, D2, D8)",
        "status": "required",
        "value": null
      },
      "seeds": {
        "note": "training seeds evaluated (protocol: 2026 preliminary; 1234, 2026, 3407 confirmatory)",
        "status": "required",
        "value": null
      },
      "startup_timeout_s": {
        "note": "worker start timeout (D4)",
        "status": "required",
        "value": null
      },
      "strict_comparison": {
        "note": "whether protocol comparisons require identical recorded limits (D5)",
        "status": "required",
        "value": null
      },
      "targets": {
        "note": "targets of the campaign among host, thor, orin, tiger",
        "status": "required",
        "value": null
      },
      "timing_boundary_id": {
        "note": "timing boundary identifier recorded by that profile (D1)",
        "status": "required",
        "value": null
      },
      "timing_profile": {
        "note": "timing profile of the protocol: supervised_service_v1 or supervised_service_rewarm_v1 (D1)",
        "status": "required",
        "value": null
      },
      "warmup": {
        "note": "{salt, k, dataset_root_sha256 (the sealed dataset root)} of the warm-up roster (decision 11)",
        "status": "required",
        "value": null
      },
      "worker_rlimits": {
        "note": "worker resource limits {nofile, fsize_mb, as_mb, cpu_s} (D4)",
        "status": "required",
        "value": null
      }
    },
    "document": null,
    "version": "required"
  },
  "record_id": "protocol-256-sealing-record",
  "schema": "passage-test-admission/v1",
  "split": "test",
  "status": "draft"
}
```

To seal it, the author fills every decision and sets its status to `sealed`, attaches the sealed protocol document
and the evidence files of section 4, sets the record status to `sealed`, and adds the approval with
`record_core_sha256` of the completed record. `passage-eval admission-check` then reports which gates remain `RED`.
