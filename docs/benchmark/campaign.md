---
title: Campaign tooling
icon: material/playlist-check
---

# Campaign tooling `passage_eval.campaign`


## 1. Executive summary

A campaign runs the procedures of a sealed plan over an admissible roster, in lots of at most 1,024 tasks, and
aggregates the scored lots per procedure. This guide describes the supervised
campaign for the historical 1,024-scene roster S at 256×256 with training seed
2026 and the frozen Seg and Reg candidates. Physical A* supplies a host-side
reference for that plan. The [paper results](candidates.md) cover a separate
direct timing campaign on the full test split with a timed same-device CPU A*
reference. The supervised harness pins each plan and opens test only when its
admission record is all green.

Three mechanisms make 32-scene lots feasible within a lease or timeout:

1. **Warm requests only.** Each procedure of a lot runs as one supervised *campaign session*: fresh workers, the
   untimed warm-up on disjoint validation inputs, then one warm request per task. There is no cold pass. The cold
   start is reported per session (worker load time per role and the first warm-up request).
2. **Dynamic request admission.** A session engages a request only while its window still holds the request's
   budget, grace, kill wait and persistence; otherwise it closes cleanly, and its remaining tasks are carried, in
   order, into the next window. The worst-case bound no longer refuses a plan; the plan must instead hold the
   observed lot time three times over and let every procedure engage at least one request.
3. **Automatic R7 relaunch.** A killed, invalid or interrupted lot is archived, never scored, and relaunched
   identically after a drain proof and a lease check, at most twice; then it is split in two. A single task that kills
   its lot twice is a candidate failure, and more than 5% of relaunched lots stop the campaign.

Section 2 describes the steps and commands, section 3 the plan, section 4 the windows and sessions, section 5 the
receipts, resume and R7 relaunch, section 6 the aggregation, section 7 the rehearsal receipt and the operator recipe,
and section 8 the limits. The test split is opened only under the [test admission gate](test-admission.md).

> **Status.** The R7 settings are effective only when identically sealed in the plan and admission record. Lease,
> lot size, overheads and margin are also sealed in the `batching` decision.

## 2. Steps and command

```bash
python -I -m passage_eval.campaign partition --parent ADMISSIBLE.json --parent-file-sha256 HEX \
  --parent-roster-sha256 HEX --out-dir partition --batch-size 4
python -I -m passage_eval.campaign check-plan --plan plan.json --plan-sha256 HEX
python -I -m passage_eval.campaign run --plan plan.json --plan-sha256 HEX [--stop-after-batches N]
python -I -m passage_eval.campaign relaunch-batch --plan plan.json --plan-sha256 HEX --batch 3 [--unit 000003.1] \
  --operator NAME --reason TEXT --reason-class infrastructure
python -I -m passage_eval.campaign aggregate --plan plan.json --plan-sha256 HEX --procedure seg --output seg.json
python -I -m passage_eval.campaign rehearsal-receipt --plan plan.json --plan-sha256 HEX --evidence-dir EVIDENCE \
  --aggregate seg=rehearsal/seg.json --aggregate reg=rehearsal/reg.json --output rehearsal/receipt.json
```

**Table 1 — Campaign steps.**

| Step | Module | What it does |
|---|---|---|
| partition | `campaign.partition` | contiguous children of at most 1,024 tasks of the admissible roster, derived by `io.roster.derive_roster`: each task keeps its `parent_rank` and its parent geographic block; a parent with a task without block is refused; no scene is read |
| plan | `campaign.plan` | reads the pinned plan and every control file it binds (section 3); `check-plan` prints the admission sizing, the worst case (information only) and the memory of one lot; runs nothing |
| run | `campaign.runner` | per lot and window, the procedures in the paired order of the lot; each is one campaign session under bwrap with the re-warm timing profile, the warm-up roster and every sealed limit, then its `score-roster` (section 4); receipts and R7 of section 5 |
| aggregate | `campaign.aggregate` | re-verifies every completion, then aggregates the warm sessions of one procedure (section 6) |
| rehearsal-receipt | `campaign.rehearsal` | builds the `passage-native-chain-rehearsal/v1` receipt of a qualifying validation rehearsal (section 7) |

## 3. The sealed plan `passage-campaign-plan/v1`

`read_plan(path, sha256)` refuses a plan whose bytes differ from the pin and any unknown or missing key at any level.
It reads only control files and lists no directory.

**Table 2 — What the plan seals.**

| Key | Content and checks |
|---|---|
| `split`, `admission` | `validation`, or `test` with `--allow-test-split` and an all-`GREEN` admission record (`record`, `record_sha256`, `evidence_dir`); an admitted plan must repeat the record's sealed decisions, rosters and harness, use the `verified-candidate-run` purpose and pin the record's digest in `protocol_record_sha256`; Table 3 lists the candidate identities it repeats |
| `target`, `edge_transport_evidence` | `host`; an edge target raises `EdgeTransportNotQualified` unless an admission record with a `GREEN` G3 for that target is named, and the runner still refuses it: no remote transport is implemented |
| `rosters`, `reference` | admissible roster (file and `roster_sha256`), source roster digest, reference admission chain (file and `chain_sha256`); the chain must name the admissible roster, the source and the plan's reference policy, cohort and search limit, and its counts must equal the admissible roster's derivation |
| `partition` | directory, manifest digest and batch size; re-derived from the admissible roster |
| `budget`, `timing` | complete budget; a re-warm timing profile and its boundary identifier; `deadline_grace_ms`, `hard_timeout_ms`, `startup_timeout_s`, `max_restarts` (at least the lot size, so that the restart cap never decides the status of a task), worker `rlimits` |
| `warmup`, `order` | validation warm-up roster built against the admissible roster on the same pinned root, its salt and `k` (each warm pass replays the `k` inputs); paired order `per_batch` with its schedule seed and procedures |
| `batching` | the window `lease_or_timeout_ms`; the measured overheads `fixed_overhead_ms`, `per_seed_overhead_ms`, `per_request_overhead_ms` and `persistence_ms`; the observed warm request time of every procedure `observed_request_ms`; `safety_margin` (at least 3). Table 4 gives the two checks that replace the worst-case refusal |
| `procedures` | for `astar`, `seg`, `reg`: pinned spec (cell, seeds, budget, reference policy, latency settings and split equal to the plan), mode, frozen decoder entry for Seg and Reg, decoder, planner, environment and split procedure pins, checkpoint (equal to `final-candidates.json` as committed at the plan's harness commit) and bundle digests |
| `harness`, `aggregation`, `seeds` | interpreters, commit and source hash (the running harness must have both, and every scored manifest must record the source hash); censoring rule identifier and minimum warm requests; one training seed per plan (one campaign per seed), which must be the training seed of the frozen checkpoint (2026) when Seg or Reg runs |
| `infrastructure_relaunch_policy`, `purpose`, `protocol_record_sha256` | the author's relaunch policy, recorded with each relaunch; run purpose; digest of the protocol record followed |

**Table 3 — Candidate identities that an admitted test plan repeats from its record (gate G6).**

| Procedure | Plan field | Must equal |
|---|---|---|
| `seg`, `reg` | `predictor`, `decoder`, `checkpoint_sha256` | the record's `candidates.<name>` entries |
| `seg`, `reg` | `bundle_sha256` | the record's `candidates.<name>.targets.<plan target>.bundle_sha256` |
| `seg`, `reg` | `decoder_allow_sha256` | exactly `[candidates.<name>.decoder_tree_sha256]` |
| `astar` | spec `inference` | a harness planner entry of `admission.CANONICAL_PLANNER_ENTRIES` (the package of the frozen decoders) |
| `astar` | `planner_allow_sha256` | exactly `[decoder_tree_sha256]` of the record, the tree of that package |

**Table 4 — Dynamic admission checks of a plan (`protocol.batching.admission_sizing`).**

| Check | Rule | Why |
|---|---|---|
| margin | `lease_or_timeout_ms >= safety_margin * sum_p expected_p`, with `expected_p = fixed + per_seed + (k + lot size) * (observed_p + per_request) + persistence` and `safety_margin >= 3` | R7: the window holds the lot, as observed on the rehearsal, three times over |
| sanity | for every procedure, `fixed + per_seed + roles * (startup_timeout_s + kill wait) + k * (observed_p + per_request) + reserve_p <= lease_or_timeout_ms / n_procedures`, with `reserve_p` = budget + grace + kill wait + persistence | a procedure's share of a fresh window can engage one request after a full start of its workers; otherwise a lot could be deferred forever |

The worst-case bound of `protocol.batching` (every request at its full budget, every restart the cap allows) stays
available in `check-plan` but no longer refuses a plan: for lots of 32 to 1,024 tasks with 120 s budgets it exceeds
any lease by orders of magnitude. The lot memory printed by `check-plan` estimates the label-free inputs a session holds
(`campaign.partition.lot_memory_bytes`): two copies of 21 bytes per cell, about 2.8 GB for 1,024 tasks at 256 × 256;
the scorer loads the lot again in its own process.

## 4. Windows, sessions and dynamic admission

An attempt of a lot runs in one or more *windows* of `lease_or_timeout_ms`. At the start of a window the runner checks
the lease (on the host: always true) and gives every procedure that still has tasks, in the lot's paired order, an
equal share of what remains of the window. The procedure runs one campaign session
(`python -I -B -m passage_eval.supervision run --campaign-session`) whose admission deadline is the end of its share.
The supervisor engages a request (initial warm-up, re-warm or measured) only while the window holds the request's
budget and grace (or hard timeout), the 15 s kill wait and `persistence_ms`, plus a full start for every worker it must
restart first; before its workers start it needs one start of every role and one request. Otherwise it closes the
session cleanly: the measured prefix is written as the run artifact, and the session receipt `session.json` lists the
deferred tasks in roster order. The runner carries them into the procedure's next window through a continuation roster
derived from the admissible roster (ranks and parent blocks kept), written in the attempt directory. The kill bound of
a session is the end of the window, never its share. The scoring of the window's sessions (`score-roster`, which
accepts the measured prefix a session declares) runs after the window, on the host, with its own bound. A window in
which no procedure measures a task stops the campaign: the plan's window cannot hold a session.

A session is *invalid* when a worker cannot start (initial start, fresh start of the pass or restart), when the initial
warm-up does not return `ok` for every request or loses a worker (it stops at its first failure, so it never consumes
the restart cap), or when the restart cap is reached. An invalid session writes no run artifact and charges no task.
A measured request that overruns its budget remains a penalised `timeout`; admission never turns it into a deferral.

## 5. Receipts, resume and relaunch

The output directory holds `campaign.json` (its plan), a lock, `invocations.jsonl` (one outcome per invocation),
`relaunch-log.jsonl`, `split-log.jsonl` and one directory per unit: a lot (`batch-000003`) or one of its R7 halves
(`batch-000003.0`). An attempt receives `intent.json` before any work (plan, unit, attempt, effect key, paired order,
deliberate kill), one `window-NN/<procedure>/` directory per session with its run, score, logs and continuation roster,
then `completion.json` after every session's score receipts are verified, or `failure.json`. The verification binds
each score receipt to the session roster and its measured prefix, the campaign session policy, every limit, and the
`checkpoint_hashes` that the supervisor records for every file a `checkpoint` or `bundle` keyword of the worker names:
a Seg or Reg run must have loaded the plan's checkpoint or bundle and nothing else, a planner run none. One warm series
per seed is required; a cold series is refused.

A completed unit is re-verified and skipped on the next run; `run --stop-after-batches N` pauses cleanly after N lots,
which is how a rehearsal exercises a resume without any relaunch.

**Table 5 — R7 relaunch rule as implemented (`campaign.runner`).**

| Event | Classification | Decision |
|---|---|---|
| session killed at the window end | `killed_by_window` | ambiguous attempt |
| session receipt `invalid` (a worker cannot start, failed initial warm-up, restart cap) | `invalid_session` | ambiguous attempt |
| `TransportLost`, or a lease not verified at a window start | `transport_or_lease_loss` | ambiguous attempt, never imputed to the candidate |
| runner stopped inside an attempt (found on resume) | `interrupted` | ambiguous attempt |
| deliberate kill of a validation rehearsal | `deliberate_kill` | ambiguous attempt, excluded from the 5% rule and from candidate failures |
| ambiguous attempt | — | drain proof first (not proven: stop, nothing moves, the next `run` decides again); archive intact under `relaunched/` with the SHA-256 of every file, never read for results nor scored; lease check (failed: stop); new effect key |
| first or second ambiguous attempt | — | `relaunch` identically: same roster, same paired order, every procedure |
| third ambiguous attempt | — | `split` in two deterministic halves (parent ranks `[start, middle)` and `[middle, stop)`) |
| a single task killed or invalid twice by the same procedure with green infrastructure | — | `impute_candidate_failure`: `runtime_error`, canonical penalty, latency charged at the wall budget and censored; the other procedures still run the task |
| a single task with three ambiguous attempts that are not attributable | — | stop until an operator `relaunch-batch` |
| more than 5% of the partition's lots relaunched | — | `stop_relaunch_fraction`; every later `run` refuses |
| supervisor exits nonzero, score receipt refused, no progress | harness error | `failure.json`; only an operator `relaunch-batch` (reason class `infrastructure`) retries it |

The host's drain proof scans the process table for a process that still names the attempt directory; the workers die
with their supervisor through their parent-death signal. An edge transport would replace it with its lease probe,
scope and accelerator checks; none exists in this harness. Every decision is appended to `relaunch-log.jsonl` with the
plan's relaunch policy, the drain proof, the lease check and the archive digest. Only the first complete attempt of a
unit is scored.

## 6. Aggregation

The six primary metrics are recomputed on the concatenated per-task rows of every session, in parent roster order,
through the R7 units and windows of every lot; a quantile is never an average of lot quantiles. Rows come from the
public loader `metrics.comparison.load_scored`, so a candidate failure stays in the denominator with the canonical
penalty; an R7 candidate failure enters with the same penalty and is listed under `imputed_failures`. The result
reports `n_source`, `n_excluded`, `n_admissible` and `n_scored`, the parent geographic block of every task (a null
block is refused), and a latency block of the proposed censoring rule `passage-censored-latency/v1` over every warm
request, with `n_requests`, `n_distinct_scenes`, `n_repeats`, p50, p95, p99 and `meets_min_warm_requests`. A series
counts as warm only if every session declared and performed its warm-up. `canonical_equivalence` checks, per session,
that the six metrics recomputed on its rows equal the scorer's `metrics.json`; an imputed failure has no scorer output
and is outside that check.

The aggregate also reports, per procedure, `cold_start` (number of sessions, median and maximum of each role's worker
load time and of the first warm-up request, which is one distinct input repeated once per session), `sessions`
(number of sessions and of sessions closed by admission), `restarts` (counted restarts, restarts outside the cap,
re-warms and failed re-warms) and `relaunches` (R7 decisions, deliberate kills, operator relaunches, relaunched lots,
archived attempts with their digests). Every seed is aggregated under `per_seed`; a plan has one seed, which is
repeated at the top level. The aggregate names its procedure, plan, target, seeds, budget, timing profile, harness and
the completion digest of every unit; gate G2 checks them against the record.

## 7. Rehearsal receipt and operator recipe

`rehearsal-receipt` writes `passage-native-chain-rehearsal/v1` only when the campaign outputs show every property that
gate G2 reads: a validation plan on the host; the running harness at the plan's commit and source hash with a clean
tree; Seg and Reg among the procedures, one aggregate per procedure under the evidence directory, equal to the
aggregate recomputed from the outputs, equal to the canonical scorer and with the plan's minimum of warm requests; a
declared and performed warm-up in every session and at least one re-warm; a deliberate kill relaunched under the
planned R7 rule with no archived attempt scored; a pause and a resume of at least one completed lot; no unplanned
relaunch. Otherwise it lists every missing property and writes nothing. Whether the budget, profile, procedures and
seed are the sealed ones is checked by G2 against the record.

**Recipe — run a validation rehearsal for Seg and Reg (not run in this repository: the frozen predictors are not
installed here).**

1. Commit and freeze the harness; check `git status` is clean under `src`, `scripts`, `pyproject.toml`, `uv.lock` and
   `docs/benchmark/final-candidates.json`, and that `final_decoders.py` has the frozen SHA-256.
2. Build the validation admissible roster and its reference admission chain, the disjoint warm-up roster (`k` sealed,
   16 proposed), and partition the admissible roster in lots of the size to seal (`partition --batch-size`).
3. Pre-rehearse a few validation scenes to measure `fixed_overhead_ms`, `per_seed_overhead_ms`,
   `per_request_overhead_ms`, `persistence_ms` and the warm request time of Seg and Reg on the host; write a plan with
   procedures `seg` and `reg` only, purpose `verified-candidate-run`, the sealed budget (120,000 ms proposed), the
   `supervised_service_rewarm_v1` profile, `max_restarts` at least the lot size, `safety_margin` 3 or more, and a
   window that passes `check-plan`.
4. `run --stop-after-batches 1`, then `run` again (the resume), with `--deliberate-kill 1:seg:SECONDS` (a delay
   shorter than the Seg session of lot 1) to exercise the R7 relaunch on purpose. At least one re-warm must happen: a
   lot with a task that overruns its budget triggers one. Check that `relaunch-log.jsonl` holds only the deliberate
   kill.
5. `aggregate` Seg and Reg into the evidence directory, then `rehearsal-receipt`; reference the receipt in the
   admission record and run `passage-eval admission-check`.

Any later commit, even to documentation under the tracked paths, invalidates G2: the rehearsal must then be repeated.

## 8. Limits

- Only the host target runs. The edge lane needs a qualified remote transport, which does not exist; the runner refuses
  an edge target even when its plan names evidence.
- The end-to-end test runs a synthetic planner as the `astar` procedure under bwrap on a synthetic validation world,
  with a pause, a resume and a deliberate kill; the frozen Seg and Reg predictors are not installed in this repository,
  and no real candidate ran through the campaign runner. Deferral across windows, splits and imputed failures are
  tested with stand-in sessions and with a real supervisor under a scripted admission clock.
- The admission window is a monotonic deadline of the host; a target-side supervisor would need a relative window.
- The overheads, the observed request times and the margin must be measured on the rehearsal; the runner does not
  re-measure them.
- Comparing routes between archived and scored attempts (the determinism flag of R7) is not implemented: the archives
  are hashed and kept for that analysis.
- The plan's checkpoint and bundle digests are not passed to the supervisor: they are enforced after each session,
  before the lot completes, on the digests the supervisor recorded. Weights that a worker loads through a keyword
  without `checkpoint` or `bundle` in its name are covered only by the candidate tree pin.
- One frozen checkpoint per candidate: a confirmatory campaign for seeds 1234 and 3407 needs per-seed checkpoints frozen
  in `final-candidates.json`, neither of which exists.
- Hashes prove byte correspondence, not authorship; every receipt says `verified: false`.
- The optional statistics command analyses pinned aggregates after the consistency checks of section 6; it does not
  re-read scored batches and is outside the current single-seed qualification. The aggregate retains every request
  for quality and censoring accounting, while its reported hot-latency quantiles exclude requests whose re-warm was
  incomplete and publish their count. A confirmatory family would additionally need frozen checkpoints for the three
  declared seeds, which do not exist.
