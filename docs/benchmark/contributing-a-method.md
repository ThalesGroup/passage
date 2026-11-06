---
title: Evaluate your planner
icon: material/account-plus-outline
---

# Evaluate your planner


The evaluator measures the complete procedure that emits a route, then scores that
same persisted route against an independently qualified reference. This guide
covers wrapping an existing classical, graph, policy or vision algorithm,
freezing its inputs and budgets, and producing a matched comparison report.
The [protocol](protocol.md) gives the metric definitions and how the evaluator
reads the paper. The [Benchmark overview](index.md) defines the terms used
here: ordinary input, complete request, qualified reference, cohort and
censored cost.

> **Evidence boundary**
>
> A small CPU run checks the interface. It establishes neither learned-model
> performance nor accelerator speed. Training and device measurements produce
> their own identified artifacts.

> **Official entries**
>
> The leaderboard submission service is not deployed yet. This guide covers
> local evaluation: the commands below produce local results, not official
> leaderboard entries.

## 1. Implement the complete procedure

From the repository root, use the locked environment:

```bash
uv sync --frozen
uv run --frozen passage-eval --help
```

Alternatively, activate it once with `source .venv/bin/activate` before using the
entry points directly. Keep the existing algorithm's dependencies and adapter
importable in this environment. The adapter is the complete deployed procedure:
its input conversion, inference/search, decoding, repair and fallback all belong
inside `plan`.

```python
from passage_eval.protocol.contracts import PlanRequest, PlanResult
from your_algorithm import solve

class MyPlanner:
    def plan(self, request: PlanRequest) -> PlanResult:
        route, work = solve(
            elevation=request.elevation, obstacles=request.obstacles,
            start=request.start, goal=request.goal, graph=request.graph,
            budget=request.budget, deadline=request.deadline_monotonic,
        )
        return PlanResult(path=route, work=work, precision="fp32")
```

This sketch assumes your `solve` returns a route and counters; adapt its arguments
to the real API. For a runnable external example, see Section 3.A. A plain
`module:function` can also accept a request and return `PlanResult`.

A request contains calibrated metre elevation, obstacles, endpoints, graph and
objective parameters, an optional normalized `(H,W,3)` tensor, and declared
budgets. Metadata uses a finite allowlist. Reference routes, optimum costs,
reference reachability and target-field scales remain evaluator-owned.

Return the original ordered integer `(row, column)` route without rounding or
clipping. Report search work, repair/fallback stages, precision and available
resource peaks. Configured work limits require their counters; fallback consumes
the same budget. Work keys are `expansions`, `queue_operations` and `live_states`
for the corresponding `BudgetSpec` limits. Use `stage_work` for primary, repair
and fallback subtotals; resource peaks use maximum rather than summed values.
`request.remaining_wall_ms()` reads the shared absolute deadline. Starting a new
timer for fallback would change the declared budget. Record `ok`, `no_path`,
`invalid_path`, `budget_exhausted`, `timeout`, `oom`, `runtime_error` or `cancelled`
as appropriate. Each decoder and budget defines a separate procedure.

Existing `Planner.predict_path(scene)` implementations and tensor/scene callables
remain supported. Use `inference_format: request` for the new interface.

The local in-process timer covers candidate-input preparation, inference, decoding, repair,
fallback, device completion synchronization, route snapshot, legality and physical
recosting. Device work completes before route snapshot and recosting. Persistence and
reference/geometry scoring follow outside it. Model construction is excluded;
the `cold` series is the first pass over the tasks after construction. Warmup and
warm outputs are retained and scored separately. The paper's direct timer
keeps independent host validation and recosting separate; see the
[evaluation protocol](protocol.md#5-trustworthiness-and-deployment) before
comparing timings across runners.

> **Deadline semantics**
>
> The runner cannot stop a Python or device call in the middle: the planner must
> check `request.remaining_wall_ms()` and stop itself (cooperative cancellation).
> The runner waits for each call to return before it starts the next request. A
> late return is a timeout even when it contains a legal route; the time spent
> before and after the deadline is recorded.

## 2. Freeze inputs, references and budgets

`uv run --frozen passage-eval init-spec --out outputs/method.yaml` writes a starter spec.
For a task specification on a local export of PASSAGE v1.0, use the following as a starting point.
This requires an existing exported dataset and sufficient reference-solve budget:

```yaml
method_name: bounded_exact
family: classical
architecture: dijkstra_prototype
inference: passage_eval.baselines.classical:BoundedExactPlanner
inference_format: request
inference_kwargs: {max_expansions: 10000}
resolution: 256
cost_model: energy
split: validation
task_variant: obstacles
n_samples: 1
seeds: [1234]
dataset_path: outputs/release/dataset
dataset_repo: thalesgroup/passage
dataset_revision: null
latency_warmup: 1
latency_repeats: 2
quality_repeat_index: 0
budget: {max_expansions: 10000}
reference_max_expansions: 10000
reference_policy: released_labels
results_root: outputs/passage-eval/qualified-v1
```

The local root must contain core `<N>x<N>/<split>/*.parquet` shards. An absent
local root fails; it never triggers a remote fallback. Remote access requires an
explicit repository and immutable 40-character commit SHA. `_samples` configs
are previews, not the training or evaluation population. Local revision metadata
alone does not verify file contents.

Freeze the sample roster and declare its split role before execution: development
for choosing algorithms, calibration for decoder/budget selection, validation for
declared intermediate checks and test for the final frozen pass. The cap selects
a deterministic prefix, not a random population sample. Keep source identity,
scene tensor/graph identity, geographic cluster assignments and exact-reference
qualification receipts with the cohort. Cohorts drawn from different data are
different cohorts, even when their scene counts match.

The default reference policy requires legal stored routes, stored-cost agreement
and independently solved optimality under the recorded tolerance. Admission
fails for an invalid or censored reference rather than penalizing the candidate.
For a deliberately new reference cohort, select
`reference_policy: independent_analytic_reference` and provide
`reference_cohort_id`. This recomputes exact references while retaining source
label defects; it creates a different cohort and does not repair training labels.

For learned methods, declare actual checkpoint files through `inference_kwargs`
or `inference_kwargs_by_seed`; missing files fail admission and their hashes are
recorded. Seeding inference does not retrain a checkpoint or prove determinism.
Batch sizes greater than one need their own supported, scored procedure; this
complete-request runner currently admits batch size one.

The run manifest binds the selected specification, procedure, candidate entry-point
source bytes and loaded code, checkpoints, evaluator source, admitted input hash
and reference identities. Include non-entry helpers and native runtime dependencies
using `runtime_bundle_manifest`, a JSON file with schema
`passage-runtime-bundle-v1` and a nonempty `files` list of `{path, sha256}` entries.
Paths resolve relative to the manifest; checksums must match readable files.
Record calibration decisions, hardware/runtime and actual precision.

## 3. Run, rescore and compare

```bash
uv run --frozen passage-eval conform --spec outputs/method.yaml --output outputs/check
uv run --frozen passage-eval run --spec outputs/method.yaml --output outputs/run-a
uv run --frozen passage-eval score --run outputs/run-a
uv run --frozen passage-eval compare --score outputs/run-a/seed-1234 --score outputs/run-b/seed-1234 --output outputs/comparison.json
uv run --frozen passage-eval report --compare outputs/comparison.json --output outputs/report.md
```

Create `run-b` with a second frozen method specification on the same tasks before
comparing. `bench --spec outputs/method.yaml` is a convenience entry point: it uses the
same measured-route workflow and writes a `Submission` record and a sidecar to its
results folder.

The conformance receipt identifies CPU checks for ordinary-input isolation,
terminal failure penalties, absolute deadlines, accumulated stage work and
malformed coordinates. It also checks an analytic graph and executes one admitted
candidate task. Its passed status certifies that bounded interface check, not
device synchronization, population label validity or candidate optimality.

Each run preserves its manifest, terminal `requests.jsonl`, numeric route bytes
and offsets in `paths.npz`, reference identities, `per_sample.npz`, `metrics.json`
and `score_receipt.json`. Arrays need no object pickle. Every declared task/repeat
has one terminal record; failures and route/status divergence remain in the
latency denominator. The frozen warm `quality_repeat_index` selects headline
quality; other repeats retain their own scored quality and latency summaries.

Comparisons require identical task, input, graph, reference and source identities.
All-task regret applies the disclosed failure cost multiplier of one million;
a failed request consequently has regret 999,999. Reports also show each method's
valid-route denominator and the both-valid geometry intersection. Empty
intersections are explicit. Connected paths and legal traversable routes have
separate rates. Geometry measures agreement with one tie-selected reference.

Geographic intervals resample whole declared geographic/source-footprint blocks
and recompute the selected statistic for each paired draw. Missing block identities
produce an unknown uncertainty status. Seed variation remains descriptive.
`compare --odd-spec bins.json` accepts a frozen joint Operational Design Domain
(ODD) stratification specification; missing support remains visible. Occupied bins
do not establish independence, representativeness or a causal geographic effect.

### 3.A. Run the external CPU example

The checked-in breadth-first-search adapter implements its own algorithm and
ordinary-input interface. The fixture producer writes two analytic Parquet tasks.
Use an unused `outputs/eval-example` destination:

```bash
uv run --frozen python -m examples.make_fixture
uv run --frozen passage-eval conform --spec examples/external_algorithm.json --output outputs/eval-example/conformance
uv run --frozen passage-eval run --spec examples/external_algorithm.json --output outputs/eval-example/bfs
uv run --frozen passage-eval score --run outputs/eval-example/bfs
uv run --frozen passage-eval run --spec examples/exact_algorithm.json --output outputs/eval-example/exact
uv run --frozen passage-eval score --run outputs/eval-example/exact
uv run --frozen passage-eval compare --score outputs/eval-example/bfs/seed-1234 --score outputs/eval-example/exact/seed-1234 --output outputs/eval-example/comparison.json --bootstrap-resamples 100
uv run --frozen passage-eval report --compare outputs/eval-example/comparison.json --output outputs/eval-example/report.md
```

The exact example uses the bounded CPU Dijkstra prototype. These commands check
adapter interoperability, failure retention and persisted artifacts. Two tiny
analytic tasks provide no real-terrain generalization or accelerator-speed evidence.
Replace their dataset and procedure specifications with a frozen real cohort for
scientific comparisons; keep the example receipts identified as fixture results.

### 3.B. Attribute improvement to algorithm components

A learned prediction claim needs more than a learned-versus-exact comparison.
Evaluate direct decoding, bounded repair with the learned guide, the same repair
machinery and budget with a prediction-free guide, an explicitly named exact
fallback, and exact and budgeted classical procedures. Reuse checkpoints across
decoder variants and charge every stage's search work and latency. An oracle guide
belongs to a separately labelled diagnostic and cannot represent ordinary input.

Declare `component_control` in each specification when producing the attribution
table. Its finite fields are `variant`, `guide`, `checkpoint_id`, `machinery_id`
and `primary_learned`. Supported variants are `direct`, `bounded_repair`,
`prediction_free`, `exact_fallback`, `exact_classical`, `budgeted_classical` and
`oracle_diagnostic`; guides are `learned`, `prediction_free`, `none` and `oracle`.
Learned variants need a checkpoint identity; bounded repair, prediction-free
repair and exact fallback need a machinery identity. Exact fallback cannot be
declared a primary learned result. The report
checks repair/prediction-free machinery and budgets, retains incomplete work
reporting, and marks missing controls rather than inferring a complete experiment.

Compare complete procedures on the same device and workload. Show legal-route
rate, all-task regret, expansions/queue/live-state work, fallback frequency and
latency together. Missing counters do not mean zero work. Stage spans help explain
the procedure but sums of stage medians do not reconstruct total latency.

## 4. Audit data and prepare notebooks

```bash
uv run --frozen passage audit --dataset-path dataset --resolution 64 128 256 512 1024 2048 4096
uv run --frozen passage analyze --dataset-path dataset --staging-root outputs/notebook-export --execute
```

The audit is bounded by default and reports exact-solve censoring. Notebook
execution performs a full four-split metadata census and bounded classical
checks. For a clearly labelled smoke run, add `--census-limit 4`,
`--classical-budget 1` and `--max-expansions 128`.

Analysis exports four notebooks per resolution (`dataset`, `display`, `metrics`
and `passage_eval_metrics`), a copy of every source notebook under `notebooks/`,
and the five dataset studies of `paper/notebooks/`, exported to
`notebooks/paper/`. The exported per-resolution notebooks drop the leading
underscore of their source templates (`_dataset.ipynb` becomes `dataset.ipynb`).

This command prepares notebooks in separate staging. Its manifest records
notebook readiness, Parquet staging and publication admission separately. Executed cells,
report/figure outputs and producing identities are checked before reuse. Snapshot
staging uses read-only hardlinks or reflinks when available and fails rather than
silently copying the full dataset; an ordinary copy requires explicit API opt-in
and adequate storage. No command in this sequence uploads or launches training.
