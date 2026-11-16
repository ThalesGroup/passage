# A small classical candidate package

This complete CPU procedure runs unweighted eight-neighbour breadth-first search
on permitted obstacle cells. It illustrates package identity and execution, not
weighted-cost optimality or model quality. It has no learned weights, no training
seed and no recorded benchmark results.

Start with the [registration and operator guide](../../docs/benchmark/candidate-registration.md)
for the full input-only, rootless execution, scoring and Space workflow. This
folder contains the runnable `mini_planner:MiniPlanner`, editable descriptor
`source.json`, actual `uv.lock`, image recipe and license. The lock contains no
additional third-party candidate dependencies. PASSAGE protocol types and NumPy
come from the separately identified common worker runtime; `uv sync` in this
folder alone does not install that runtime.

## Freeze a working copy

From the core repository root, activate its configured Python 3.12 environment:

```bash
source .venv/bin/activate
python -m unittest discover -s examples/candidate-package -p 'test_*.py'
export PASSAGE_WORK="$(mktemp -d -t passage-example.XXXXXXXX)"
cp -R examples/candidate-package "$PASSAGE_WORK/package"
passage-eval candidate freeze "$PASSAGE_WORK/package/source.json" \
  --out "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate validate "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate register "$PASSAGE_WORK/package/candidate.json" \
  --catalog "$PASSAGE_WORK/catalog.json"
```

The descriptor's `sha256: null` entries are instructions for `freeze`, not fake
hashes. The sealed manifest contains hashes of the actual files. `candidate.json`
is generated locally and not shipped. Freeze refuses an existing destination;
use a fresh copy for another version. Registration does not execute the planner.
The pinned dataset revision is a concrete starting identity; match it to the
operator's real prepared index before freezing a measurement version.

## Supply an already built runtime

`Containerfile` wraps an operator-provided common worker image; it does not fetch
PASSAGE or weights. Set `PASSAGE_RUNTIME_IMAGE` to the actual immutable common
image reference before building on an available image builder:

```bash
docker build --pull=false --network=none \
  --build-arg RUNTIME_IMAGE="$PASSAGE_RUNTIME_IMAGE" \
  -f examples/candidate-package/Containerfile \
  -t local/passage-example-bfs:0.1.0 examples/candidate-package
```

The image must contain Python 3.12, NumPy and the installed
`passage_eval.campaign.candidate.worker`. The candidate is mounted separately at
`/candidate`, so no source or weight file is copied into this image. Record its
actual immutable reference and execution receipts. If adding the final image
reference to an already registered manifest, give the changed package a new
version. A local tag is not an immutable digest and is not accepted as a non-null
manifest `image.reference`.

A systemd-blocked Docker host cannot run this build/preparation command unchanged;
an operator-provided rootfs receipt can still be used by the rootless `runc`
execution path documented in the guide. No image build or publication is claimed
by this example.
