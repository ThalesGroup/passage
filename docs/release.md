---
title: Release 1.0.0 and installation
icon: material/package-check
---

# Release 1.0.0 and installation

PASSAGE **1.0.0 is the first official code release**. It includes the
generator, evaluator and common candidate package interface: freeze, validate,
register, CPU/CUDA worker execution, rootless OCI sessions and host scoring.
The dataset has its own `v1.0` identity. A code version does not change the dataset
revision declared in a candidate manifest.

This page describes source installation and locally built, hash-checked artifacts.
Obtain the source from [`ThalesGroup/passage`](https://github.com/ThalesGroup/passage)
and record its exact commit; use the `v1.0.0` tag when available. Model access and
runtime-image availability are separate prerequisites. No package-index
installation or public CUDA image is assumed.

## Install the reviewed checkout

Use Python 3.11.5 or later and [uv](https://docs.astral.sh/uv/). The supplied CPU
candidate example requires Python 3.12, so use that minor for the full example.
Obtain the reviewed 1.0.0 source checkout or complete source snapshot and record
its commit or archive SHA-256. From its root:

```bash
uv sync --frozen --python 3.12
uv run --frozen python -c 'from importlib.metadata import version; assert version("passage") == "1.0.0"'
uv run --frozen passage --help
uv run --frozen passage-eval candidate --help
source .venv/bin/activate
```

The root `uv.lock` pins the checkout environment. An extracted complete source
snapshot must include that lock and `examples/candidate-package/`. A core wheel
provides installed modules and command-line entry points; obtain the example
source separately from the same reviewed source tree. Do not assume a wheel or
source distribution contains every repository example.

## Build fresh core artifacts

Maintainers build the reviewed source into a fresh output directory. The build
requires uv and access to the declared setuptools build dependency, either already
available or through the configured dependency source. It does not require Torch,
a CUDA device, training repositories or a runtime image.

```bash
export PASSAGE_DIST="$(mktemp -d -t passage-1.0.0-dist.XXXXXXXX)"
uv build --out-dir "$PASSAGE_DIST"
(
  cd "$PASSAGE_DIST"
  sha256sum passage-1.0.0-py3-none-any.whl passage-1.0.0.tar.gz > SHA256SUMS
  sha256sum -c SHA256SUMS
)
```

Keep the source identity, build/check logs and checksums with the artifacts. The
checksum file detects changed bytes; compare it with the maintainer's reviewed
record when receiving an artifact. Merely generating a new checksum for an
unknown wheel does not establish its provenance.

Historical private wheels include files named `passage-1.0.0-py3-none-any.whl`,
and private runtime/measurement receipts also use 1.1.0. Preserve their filenames,
versions and hashes. The freshly built official candidate must be identified by
its new source identity and artifact hash; neither an old same-version filename
nor renaming a 1.1.0 wheel makes it this release.

## Install the exact wheel in a separate environment

Set `PASSAGE_DIST` to the directory containing the supplied wheel and its reviewed
`SHA256SUMS`. Verify it before installation, then use a fresh environment:

```bash
(cd "$PASSAGE_DIST" && sha256sum -c SHA256SUMS)
export PASSAGE_INSTALL="$(mktemp -d -t passage-1.0.0-install.XXXXXXXX)"
uv venv --python 3.12 "$PASSAGE_INSTALL/venv"
uv pip install --python "$PASSAGE_INSTALL/venv/bin/python" \
  "$PASSAGE_DIST/passage-1.0.0-py3-none-any.whl"
source "$PASSAGE_INSTALL/venv/bin/activate"
python -c 'from importlib.metadata import version; assert version("passage") == "1.0.0"'
passage-eval candidate --help
python -m passage_eval.campaign.candidate.worker --help
```

This installs the exact core wheel and resolves its declared runtime dependencies.
It does not reproduce the root lock or the private ML snapshot. Use the frozen
checkout route when the root lock is required; retain an inventory of any
separately installed runtime.

## Run the CPU package example

With the installed environment active, work from the matching complete source
root and follow the [small classical example](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/README.md).
These commands check packaging without a model, GPU, Docker or data download:

```bash
export PASSAGE_WORK="$(mktemp -d -t passage-example.XXXXXXXX)"
cp -R examples/candidate-package "$PASSAGE_WORK/package"
passage-eval candidate freeze "$PASSAGE_WORK/package/source.json" \
  --out "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate validate "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate register "$PASSAGE_WORK/package/candidate.json" \
  --catalog "$PASSAGE_WORK/catalog.json"
```

Expect a validated manifest and a catalog entry. The example is unweighted
breadth-first search, with no learned weights or claimed benchmark result.

For a local worker check, first export a matching input-only NPZ using the
[operator guide](benchmark/candidate-registration.md#operator-export-input-only-arrays-and-retain-the-matching-cohort).
Keep its cohort and reference files on the host. Set `PASSAGE_INPUTS` to that NPZ
and use a new output directory:

```bash
python -m passage_eval.campaign.candidate.worker \
  --manifest "$PASSAGE_WORK/package/candidate.json" \
  --inputs "$PASSAGE_INPUTS" --output "$PASSAGE_WORK/run-cpu" --device cpu
```

The direct worker writes `journal.jsonl` and `run.json`; it is a local execution
check without the container receipt. For a measured rootless session, prepare a
compatible image/rootfs and follow the guide's OCI run, receipt binding and
independent host scoring steps. The image must contain this exact core and match
the candidate's Python minor. The
[private ML runtime recipe](https://github.com/ThalesGroup/passage/blob/main/containers/candidate/README.md)
additionally needs its authorized ARM64 environment snapshot and pinned base.
CUDA execution requires compatible drivers, device exposure and an explicitly
allowlisted GPU; a declared CUDA backend alone proves no execution.

## Check locally before the release decision

Maintainers should retain results for the candidate/evaluator tests, builder and
example tests, version/lock agreement, wheel and source-distribution metadata,
installation outside the checkout, and the freeze/validate/register/worker/score
route. With the existing development and documentation dependencies available,
these commands provide the focused checks:

```bash
uv run --frozen pytest tests/candidates tests/passage_eval
uv run --frozen python -m unittest discover -s containers/candidate -p 'test_*.py'
uv run --frozen python -m unittest discover -s examples/candidate-package -p 'test_*.py'
uv run --frozen ruff check src/passage-eval/campaign/candidate tests/candidates
uv run --frozen --group docs mkdocs build --strict --site-dir "$PASSAGE_DIST/site"
```

Confirm the actual test paths in the reviewed source if its layout changes.
Record failures and unavailable checks; these commands are a checklist, not a
claim that a particular artifact passed. Retain runtime configuration, platform
manifest and OCI index digests separately when an image is built.

The release owner must review these artifacts and approve the final source
synchronization, tag and publication. Before that decision, verify public links
against the final exported tree and check artifact availability and access.
Model repositories, runtime images and the private leaderboard Space have
separate access/publication gates. Its active SEG/REG view displays 27 TEST and
three separate VALID pilot observations as `measured_unadmitted`; historical
observations retain their provenance. They are separate from the repository's empty checked-in
TEST entry table and do not become admitted results through a core release.
