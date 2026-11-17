---
title: Official evaluation
icon: material/certificate-outline
---

# Official evaluation

This page is the Thales runbook that turns a [submission](candidate-registration.md)
into an official leaderboard result. It lists the tasks of the PASSAGE operator,
in order, with the condition that closes each one. Contestants can read it to know
what happens to their package. They have nothing to do here.

## :material-account-group-outline: Roles and results

- **The contestant** owns the package: its code, weights, locks, recipe,
  declarations and rights.
- **The operator** is a Thales team member who prepares the scenes and
  references, runs the package, scores the routes and reviews the evidence.
- **The reviewer** is a second Thales team member who checks the evidence and
  signs the admission of a verified result.

A result is **measured** after steps T1–T6 and **verified** after step T7. The
[leaderboard](leaderboard/index.md#entry-status) explains both statuses. Every step
answers in the contestant's submission request, so its history stays in one
place.

## :material-wrench-cog-outline: One-time setup

Verification needs three prerequisites. None exists yet, so every current row is
*measured*.

1. **Operator key.** Create an Ed25519 key on the reviewer's machine and pin its
   public half in the `allowed_signers` file of the private results archive with
   a reviewed commit. The tools never generate a key.
2. **Admission policy.** Sign one admission policy for the leaderboard task
   profile. It names the reviewer, the qualified runtime of each target (lane,
   observed runtime, candidate and host-verifier commits) and the timing boundary.
3. **Trust pin in the leaderboard.** Add the same public key to the leaderboard's
   `data/native-trust.json` with a reviewed commit, so that the viewer accepts
   signed results.

The results archive's operator guide gives the exact commands and file formats.

A candidate **family** that the leaderboard has not published before also needs
a reviewed change to its publication validators. Today they accept `seg` and
`reg` candidates and the A* reference.

## :material-format-list-checks: Steps for each submission

### T1 · Intake

Read the request and the package card. Check that:

- the repository is reachable at the stated revision, and its licence lets
  Thales run the package and publish its results;
- each listed manifest exists and its `manifest_sha256` matches;
- the package declares ordinary inputs only, the leaderboard dataset, task and
  `wall_time_ms: 5000`, three seeds (or none for a classical planner), and its use
  of the test split;
- accelerated targets come with an ONNX export of the network and the decoder
  source;
- the local check covered 1,024 validation scenes and completed.

*Done when* the request is accepted, or answered with the list of changes needed.

### T2 · Freeze and register

Mirror the repository at the stated revision into the operator area. Validate and
register each seed's manifest in the operator catalog:

```bash
passage-eval candidate validate "$PACKAGE/candidate.json"
passage-eval candidate register "$PACKAGE/candidate.json" --catalog "$OPERATOR/catalog.json"
```

Registration refuses an identity reused for different bytes. Validation and
registration never import the planner.

*Done when* every seed is registered and its identity matches the request.

### T3 · Prepare the runtimes

- **Common runtime.** Build the image from the package recipe with the core wheel
  of the qualified release, then export its root file system:

    ```bash
    python -m passage_eval.campaign.candidate.oci prepare \
      --image "$CANDIDATE_IMAGE" --output "$OPERATOR/rootfs"
    ```

    The receipt binds the exported archive and tree hashes. Record the image
    identity with the receipt.

- **Native runtime for each accelerated target.** Compile the ONNX export for the
  target runtime: TensorRT on the Jetson devices, OpenVINO on Tiger Lake. The
  decoder stays on the target CPU. Run one validation scene through each native
  build and the CPU reference, and record the export validation report.

*Done when* every target has an identified runtime and a passing export check.

### T4 · Qualify on validation

Repeat the contestant's check on the operator's own prepared validation cohort,
first in the common runner on a workstation, then on each target:

```bash
python -m passage_eval.campaign.candidate.inputs \
  --index "$OPERATOR/prepared/validation/index.json" \
  --output "$OPERATOR/inputs/validation.npz" --count 1024
python -m passage_eval.campaign.candidate.oci run \
  --rootfs "$OPERATOR/rootfs/rootfs-receipt.json" --manifest "$PACKAGE/candidate.json" \
  --inputs "$OPERATOR/inputs/validation.npz" --output "$OPERATOR/runs/validation-cpu" \
  --device cpu --timeout-seconds 7200
python -m passage_eval.campaign.candidate.binding --run "$OPERATOR/runs/validation-cpu"
python -m passage_eval.campaign.candidate.scoring --run "$OPERATOR/runs/validation-cpu" \
  --index "$OPERATOR/inputs/validation.cohort.json" --target workstation-cpu \
  --output "$OPERATOR/rows/validation-cpu.json"
```

Only the input-only `.npz` enters the runner; never bind the host index or the
cohort file. Compare the scores with the contestant's local check. You may show
the workstation row in the leaderboard's *Validation pilot* panel; it stays
separate from the test table.

*Done when* all 1,024 scenes complete on every target, with no unexplained
difference from the contestant's check. Send the qualification summary to the
contestant.

### T5 · Measure the test split

For each seed and each target, open one persistent batch-one session, make 16
warm-up calls on one separate validation scene, then time one request per test
scene over the 40,000 scenes. The timer runs from raw in-memory arrays to the
returned route; scoring stays outside it. Record the power profile before and
after each session. Run one CPU A* reference session on each target if none
exists for the same runtime.

Keep the journals, the session records, the runtime receipts and the cleanup
evidence. A normal exit alone does not prove a complete session: check the
completion counts, the unchanged package, input and runtime identities, and the
drained namespaces.

*Done when* every requested target and seed has a complete session.

### T6 · Score and review

Score every session on the host against the stored references. The scorer
rechecks legality, recomputes each cost and turns invalid reported successes into
failures. Then review each session:

- all 40,000 scenes are present, in the cohort order;
- the status counts, latencies and the six metrics are consistent with the
  qualification run;
- no reference data entered the candidate namespace;
- the identities in the scores match the registered package and the runtimes.

Send the results to the contestant before publication.

*Done when* each session has a reviewed score. The result is now **measured**.

### T7 · Verify

The reviewer checks the executed artifacts of each session against the package:
weights or exports, decoder, engine, dependency lock and preprocessing. They then
sign one artifact attestation per session under the admission policy. The
results archive's publication tool turns each signed session into a verified
receipt. The archive's operator guide describes both commands.

*Done when* every session has a verified receipt. The result is now **verified**.

### T8 · Publish

1. Export the public rows into the leaderboard's `data/` files: one row per target
   and seed, with its evidence identities. Private journals, references, weights
   and host paths never enter them.
2. Run the leaderboard checks, then rebuild its single-document page:

    ```bash
    npm test && npm run build:ui && npm run check:bundle
    python static_page.py
    python -m unittest test_server test_static test_browser
    python static_page.py --check
    ```

3. Inspect the new rows in the local page, then push the reviewed commit, which
   redeploys the leaderboard. Check the served page.
4. Close the request with the row identities, and archive the private evidence
   with them.

*Done when* the rows are online and the request is closed.

## :material-backup-restore: Corrections and withdrawal

A changed package is a new version and goes through T1–T8 again; earlier rows
keep their identities and remain visible. To correct a published row, publish
the corrected result as a new row and record the correction in the earlier row's
evidence limitations; never change its values in place. A contestant can ask for
withdrawal at any time: remove the row with a reviewed commit and keep the
private evidence.

## :material-alert-outline: Current limits

- No verified result exists until the one-time setup is complete.
- The common runner is a pilot. It runs `cpu` and `cuda` only, does not enforce
  resource limits, and records `strict_admission: false`. Its isolation is
  described on the [rootless OCI runner](candidate-oci.md) page.
- The native runtimes of the edge targets rely on Thales-internal inference code
  and on vendor SDKs, so each new candidate's native integration is done by
  Thales and checked at T3.
