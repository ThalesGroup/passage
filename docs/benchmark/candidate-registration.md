---
title: Register and run a candidate package
icon: material/package-variant-closed-check
---

# Register and run a candidate package

A candidate package freezes the complete procedure that receives ordinary scene
inputs and emits a route. The author supplies source, optional weights, a dependency
lock and an image recipe. The operator registers its immutable identity, runs it
with input-only arrays, scores the persisted routes on the host, then exports a
reviewed measurement to the Space. Registration is local bookkeeping; it does not
publish a package or admit a leaderboard result.

The first official core release, **1.0.0**, includes this common interface.
Public source links below target GitHub `main`; record the exact source commit
or release tag used for an execution.

This guide covers `passage-candidate-package/v1` and the exploratory
`pilot-validation-r256/v1` profile. Start with the runnable
[classical example](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/README.md), then use the
operator steps below for a real prepared validation cohort. The
[ordinary-input protocol](protocol.md) and [planner implementation guide](contributing-a-method.md)
explain the graph, metric definitions and general evaluator interface. This
package runner has its own explicit timing boundary.

## Prerequisites and roles

Install the exact 1.0.0 core from the reviewed checkout or a freshly built,
hash-checked wheel using the [release and installation guide](../release.md).
It supplies the `passage-eval candidate` commands and
`passage_eval.campaign.candidate` modules. Activate that environment before the
commands below. The candidate's declared Python
minor must match the worker; the example declares Python 3.12. Package validation
and registration never import the planner.

The author owns the deployed procedure and its provenance. The operator owns the
prepared input index, reference paths, execution supervision, independent host
scoring and result review. Reference routes, optimum costs, reference reachability
and supervision fields must remain outside the candidate package and mounted
inputs. Training and calibration use TRAIN data; declare any previous public TEST
use, including tuning or exploratory exposure.

For container execution, prepare a compatible runtime image with the common
PASSAGE worker already installed. See the
[private ML runtime build guide](https://github.com/ThalesGroup/passage/blob/main/containers/candidate/README.md) for its
snapshot-based recipe and its access prerequisites. This is a private build
kit, not a public CUDA image or an installable inventory. A compatible CPU runtime
can use the same worker without
ML dependencies added by the candidate. Image preparation requires access to an
existing local image through Docker; execution uses rootless `runc` and a retained
rootfs receipt. These are distinct capabilities.

**Runtime limit.** Image preparation needs a working builder/export route; an
operator-supplied rootfs receipt is an alternative when local Docker is unavailable.
The launcher does not enforce resource cgroups and records
`resource_cgroup_enforced: false` and `strict_admission: false`. These pilot runs do
not establish strict admission, accelerator qualification or resource isolation
beyond the recorded mechanisms.

## Author: implement and describe the complete procedure

The entry point is `module:Class`. The worker constructs the class with the sealed
`kwargs` plus `device="cpu"` or `device="cuda"`, then calls `plan(PlanRequest)`.
Return a materialized ordered route of integer `(row, column)` cells in
`PlanResult`, not a lazy iterator. The complete procedure includes preprocessing,
graph construction, inference, all policy steps, decoding, repair and fallback.

Use `request.budget` and `request.remaining_wall_ms()` across every stage. Report
`work` counters `expansions`, `queue_operations` and `live_states` when their limits
are configured, plus `stage_work` subtotals for primary, repair and fallback.
Resource peaks use maxima; operation counts include all stages. Algorithm-specific
limits belong in `kwargs`, not new budget fields. Return a terminal status from
`ok`, `no_path`, `invalid_path`, `budget_exhausted`, `timeout`, `oom`,
`runtime_error` or `cancelled`. A CUDA candidate must implement `synchronize()`;
release retained resources in `close()` when needed.

The [example module](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/mini_planner.py) implements
ordinary eight-neighbour breadth-first search with destination-only obstacles.
It is a reachability example, not an elevation-cost optimum and not a learned
method. It has no weights, training seed or fabricated results. Its lock records
only its additional dependencies; the separately identified common image supplies
PASSAGE and NumPy.

Create an editable descriptor such as
[`source.json`](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/source.json). Its artifact hashes
are `null` until `freeze` computes them from actual files. A descriptor is not yet
a valid sealed manifest. The
[JSON Schema](https://github.com/ThalesGroup/passage/blob/main/schemas/candidate-package-v1.schema.json) describes the exact
manifest fields. The Python loader additionally verifies filesystem state,
symlinks, hashes and cross-field artifact relationships.

| Field | Meaning and accepted values |
|---|---|
| `schema` | Exactly `passage-candidate-package/v1`. |
| `candidate_id`, `version`, `family` | Each matches `[a-z0-9][a-z0-9.-]{0,95}`. The registry key is `candidate_id@version`. |
| `entrypoint`, `kwargs` | Importable `module:Class` inside the sealed package and a JSON object of constructor parameters. Relative `checkpoint` and `root` parameters are resolved against the package root. |
| `python`, `backends` | Python minor `3.11`, `3.12`, `3.13` or `3.14`; a nonempty unique list drawn from `cpu`, `cuda`, `openvino`, `tensorrt`. The current worker/OCI device interface runs `cpu` or `cuda` only. |
| `seed` | Nonnegative integer or `null`. The displayed pilot requires 2026 for learned candidates; classical references use `null`. A package can validate without qualifying for that display profile. |
| `dataset` | Exact fields `id`, immutable 40-character lowercase commit `revision`, `config` such as `256x256`, `cost_model` and finite nonnegative `cost_weight`. Models are `distance`, `elevation`, `absolute_elevation`, `slope_uphill`, `energy`; `slope` is an accepted alias. |
| `budget` | Only `wall_time_ms`, `max_expansions`, `max_queue_operations`, `max_live_states`; omitted or `null` means no configured limit. Wall time must be finite and positive; work limits must be positive integers. |
| `precision`, `license` | Nonblank descriptions of actual execution precision and applicable rights. |
| `training` | Object with `split: "train"` or `"none"` and a truthy `test_usage` declaration. Add meaningful provenance: training/calibration data revisions, code, weights, label generation and selection policy. The loader permits extra provenance fields; validation does not verify their claims. |
| `image` | Exactly `recipe` and `reference`. Recipe names a declared file. Reference is `null` before a build, or an immutable `name@sha256:…` reference. The loader does not resolve a tag or authenticate an image. |
| `artifacts` | Nonempty records with exactly `path`, actual lowercase SHA-256 and `role`. Roles are `code`, `weights`, `config`, `lock`, `recipe`, `notice`; at least code, lock and recipe are required. Weights are optional. |

Artifacts must be materialized files under the package root, without symlinks,
absolute paths or `.`/`..` path segments. Declare every file required at runtime:
the OCI launcher stages only declared artifacts and the sealed manifest. Do not
rely on unlisted source modules or ambient host files. Keep private credentials,
reference labels and arbitrary host paths out of the package.

## Author and operator: freeze, validate and register

From the repository root, create a disposable copy of the example or substitute
your own complete package:

```bash
export PASSAGE_WORK="$(mktemp -d -t passage-candidate.XXXXXXXX)"
export PASSAGE_PACKAGE_ROOT="$PASSAGE_WORK/package"
cp -R examples/candidate-package "$PASSAGE_PACKAGE_ROOT"
passage-eval candidate freeze "$PASSAGE_PACKAGE_ROOT/source.json" \
  --out "$PASSAGE_PACKAGE_ROOT/candidate.json"
passage-eval candidate validate "$PASSAGE_PACKAGE_ROOT/candidate.json"
passage-eval candidate register "$PASSAGE_PACKAGE_ROOT/candidate.json" \
  --catalog "$PASSAGE_WORK/catalog.json"
```

`freeze` requires the output manifest to be beside its descriptor and refuses an
existing output file. It hashes declared files and validates the resulting
manifest without executing its entry point. Success reports `identity`,
`manifest_sha256` and `status: "validated"`. Registration reports `added: true`;
repeating it with identical bytes reports `added: false`. The local catalog stores
the manifest path, so retain that package at its registered location.

Changing source, weights, kwargs, dependency lock, recipe, image reference,
budgets or provenance changes the manifest identity. A registered
`candidate_id@version` cannot be reused for different bytes: create a new version,
freeze to a new package directory and register it. Build and pin the final image
before final registration when its reference is to be part of the sealed package.
The example's `image.reference: null` permits a local packaging check but does not
claim an immutable executed image; retain the rootfs/image execution receipts too.

## Operator: export input-only arrays and retain the matching cohort

Start from a trusted, prepared validation index with schema
`passage-training-scenes/v1`, immutable `dataset` identity and `scenes` records
containing sample IDs, source NPZ paths and their hashes. The dataset object must
match the candidate manifest, including cost model and weight. Preparation of
those host references is a separate operator step; registration does not generate
reference paths.

Set `PASSAGE_HOST_INDEX` to that existing host index. Export a bounded prefix:

```bash
python -m passage_eval.campaign.candidate.inputs \
  --index "$PASSAGE_HOST_INDEX" \
  --output "$PASSAGE_WORK/inputs/scenes.npz" --count 128
```

This reads and checks the prepared file hashes, then writes:

- `scenes.npz`: only `sample_ids`, `terrain`, `obstacles`, `starts`, `goals`.
- `scenes.cohort.json`: the exact corresponding host reference cohort, including
  source index identity and host reference file locations.
- `scenes.receipt.json`: input-file and cohort-index SHA-256 identities, split and
  population size.

Only `scenes.npz` enters the candidate namespace. Never bind the inputs directory
or the host index: the two JSON sidecars and original reference NPZ files remain
host-only. The worker checks unique string sample IDs, finite float64 terrain,
boolean obstacles, int64 endpoints, declared resolution and population size.

The helper loads one fragment into memory; it rejects fragments approaching its
2-GiB bound. At R256, 128 or 1024 scenes fit this bound. A 40,000-scene monolithic
export is not supported by this transport. `--count` selects a prefix, not a random
sample or a fragment offset; use separately prepared indices for disjoint cohorts
and retain their identities. Always score against the emitted
`scenes.cohort.json`, especially when exporting a subset.

The Python API is
`export_input_cohort(index_path, output, count=128)` from
`passage_eval.campaign.candidate.inputs`; it returns the same receipt.

## Operator: execute a rootless OCI session

OCI means Open Container Initiative. The launcher creates its own rootless
`runc` state, PID/user/mount/network namespaces, read-only rootfs and candidate
mounts, an input-file bind and writable output. It verifies the package and input
bytes again, drains the runtime and records cleanup. The network namespace has no
configured external connection. A private rootfs receipt binds the selected local
image/export and tree hashes; an image configuration ID and a registry manifest
digest are different identities and must not be relabeled.

If a compatible local image is available through Docker, prepare a new receipt:

```bash
python -m passage_eval.campaign.candidate.oci prepare \
  --image "$PASSAGE_RUNTIME_IMAGE" --output "$PASSAGE_WORK/rootfs"
```

`PASSAGE_RUNTIME_IMAGE` must name the actual compatible worker image. The example
Containerfile can wrap an already prepared immutable common runtime using
`--build-arg RUNTIME_IMAGE=…`; see its [README](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/README.md).
`prepare` inspects, creates and exports an already-local image without pulling.
When Docker is unavailable, obtain the retained rootfs directory, archive and
receipt from the operator and use their actual receipt path below. Those private
artifacts must remain together and unchanged; the receipt includes their paths.

Run CPU execution with a new, disjoint output directory:

```bash
python -m passage_eval.campaign.candidate.oci run \
  --rootfs "$PASSAGE_WORK/rootfs/rootfs-receipt.json" \
  --manifest "$PASSAGE_PACKAGE_ROOT/candidate.json" \
  --inputs "$PASSAGE_WORK/inputs/scenes.npz" \
  --output "$PASSAGE_WORK/run-cpu" --device cpu --timeout-seconds 300
```

The output directory must not already exist and must not overlap the package,
rootfs or inputs. The launcher stages only declared candidate artifacts, then
loads the image-installed worker. The CPU path hides CUDA devices. Package Python
minor and backend declarations are checked before planner construction.

For an actual CUDA-capable package, select `--device cuda --gpu-uuid
"$PASSAGE_GPU_UUID"` instead. The current launcher accepts only the operator's
explicitly allowlisted GPU UUID and reads its CDI (Container Device Interface)
configuration; it does not auto-select a free GPU. A declared `cuda` backend or a
CUDA-enabled framework build is not evidence of a successful CUDA run. The
classical example intentionally declares CPU only. GPU allocations, cumulative
training/evaluation time and campaign reserves must be approved and accounted for
before any device launch.

The per-scene boundary is
`raw_in_memory_arrays_to_emitted_path_non_supervised`: input-copy preparation,
procedure work, synchronization and materialized emitted path are timed. Model
construction, archive loading and host reference scoring are outside it. There
are no warmups in this runner. A cooperative deadline is checked again after safe
completion; late routes remain timeout failures with their observed duration.
An outer `--timeout-seconds` kill is a session guard, not a way to convert an
incomplete cohort into a successful measurement.

Keep `journal.jsonl`, `run.json` and `oci-run.json`, plus the private runtime
configuration, stdout/stderr, rootfs/image receipts and cleanup evidence. A normal
CLI exit alone does not establish a complete cohort or strict admission. Check
return code, timeout flag, completion counts, unchanged rootfs/input/package
identities and namespace drain before scoring.

Bind the host-observed container receipt after successful execution and cleanup:

```bash
python -m passage_eval.campaign.candidate.binding --run "$PASSAGE_WORK/run-cpu"
```

This adds the OCI receipt hash, image identity and rootfs receipt hash to
`run.json`. The scorer rechecks that binding before exporting it. The resulting
run hash therefore covers the container evidence as well as the worker results.
This is retained operator evidence, without a signature or admission claim.

## Operator: score on the host and export the Space row

Use the exact cohort index emitted with this input fragment and an existing
completed run directory. Set a nonprivate target slug for the actual workstation;
this example uses `workstation-cpu`. Supply the actual candidate training cost in
GPU seconds. For the untrained classical example it is zero:

```bash
python -m passage_eval.campaign.candidate.scoring \
  --run "$PASSAGE_WORK/run-cpu" \
  --index "$PASSAGE_WORK/inputs/scenes.cohort.json" \
  --target workstation-cpu --output "$PASSAGE_WORK/pilot-row.json" \
  --gpu-seconds 0
```

The host verifies journal/input identities and exact ordered cohort coverage,
recosts routes, checks legality and corrects invalid reported successes. Failed or
late routes remain failures. It writes retained `score.json` and
`scored-arrays.npz` alongside the run, and exports one finite pilot row. Inspect
metric values, all terminal status counts, observed/censored latency, cost,
selection provenance and evidence hashes. Omit `--gpu-seconds` when historical
training cost is unknown; the exported value is `null`, without implying zero.
Training cost is not total campaign
GPU usage. Retain training code, checkpoints, dataset/label-generation identities,
validation selection and budget logs separately; labels never enter the measured
candidate inputs.

The Space profile uses validation R256, seed 2026 for exploratory learned
candidates or seedless classical references, with `status: measured_unadmitted`
and `verification: null`. The scorer initially uses the candidate ID as the display
label; an operator may choose a truthful public `candidate_label` without changing
the candidate/run evidence identities. Keep versions, seeds, populations,
workstations and CPU/CUDA backends separate. A single seed establishes neither a
multi-seed conclusion nor a cross-hardware rank.

After reviewing the row, add it to `rows` in the Space's
`data/experimental-measured.json`, preserving schema
`passage-experimental-measured/v1` and relevant `limits`. Leave
`data/native-measured.json` and its 27 TEST observations untouched. The private
Space's active SEG/REG view displays those 27 TEST and three separate VALID
pilot observations as `measured_unadmitted`; historical observations retain
their original provenance. They are separate from the public repository's local
storage and empty checked-in TEST entry table. Access to that Space and its
operator guide is required for the following optional display step. Use the Space's
[operator guide](https://huggingface.co/spaces/thalesgroup/passage-leaderboard/blob/main/docs/EXPERIMENTAL_PILOT.md)
for exact display validation. In that separate checkout, run:

```bash
npm test
npm run build:ui
npm run check:bundle
python static_page.py
python -m unittest test_server test_static test_browser
python static_page.py --check
```

Inspect the local **Validation pilot** panel and retain its evidence limitations.
These steps prepare a reviewable static artifact. They do not upload results,
change Space visibility, sign admission or execute an official leaderboard service.
