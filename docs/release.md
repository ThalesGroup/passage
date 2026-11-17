---
title: Release 1.0.0 and installation
icon: material/package-check
---

# Release 1.0.0 and installation

PASSAGE **1.0.0 is the first official code release**. It includes the
generator, the evaluator and the common candidate package interface: freeze,
validate, register, CPU/CUDA worker execution, rootless OCI sessions and host
scoring. The dataset has its own `v1.0` identity; a code version does not change
the dataset revision that a candidate manifest declares.

This page installs the code from source and builds hash-checked artifacts. No
package-index installation or public CUDA image is assumed.

## :material-source-branch: Install from the release tag

Use Python 3.11.5 or later and [uv](https://docs.astral.sh/uv/). The CPU
candidate example declares Python 3.12, so use that minor to run it:

```bash
git clone --branch v1.0.0 https://github.com/ThalesGroup/passage.git
cd passage
uv sync --frozen --python 3.12
uv run --frozen python -c 'from importlib.metadata import version; assert version("passage") == "1.0.0"'
uv run --frozen passage --help
uv run --frozen passage-eval candidate --help
source .venv/bin/activate
```

The root `uv.lock` pins this environment. A wheel provides the installed modules
and command-line entry points but not the repository examples, so keep the
source tree for `examples/candidate-package/`.

## :material-hammer-wrench: Build the core artifacts

Build the checkout into a fresh output directory. The build needs uv and the
declared setuptools build dependency; it needs no Torch, CUDA device or runtime
image.

```bash
export PASSAGE_DIST="$(mktemp -d -t passage-1.0.0-dist.XXXXXXXX)"
uv build --out-dir "$PASSAGE_DIST"
(
  cd "$PASSAGE_DIST"
  sha256sum passage-1.0.0-py3-none-any.whl passage-1.0.0.tar.gz > SHA256SUMS
  sha256sum -c SHA256SUMS
)
```

Keep the source commit and the checksums with the artifacts. A checksum detects
changed bytes; it does not by itself establish where an unknown wheel came from.
A candidate image installs this wheel, so record its hash with the image.

## :material-package-down: Install the wheel in a separate environment

Verify the wheel against its `SHA256SUMS`, then install it in a fresh
environment:

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

This resolves the wheel's declared runtime dependencies, not the root lock. Use
the checkout route above when you need the locked environment.

## :material-play-circle-outline: Run the CPU package example

From the source tree, with the environment active, freeze and register the
[small classical example](https://github.com/ThalesGroup/passage/blob/main/examples/candidate-package/README.md).
These commands need no model, GPU, Docker or data download:

```bash
export PASSAGE_WORK="$(mktemp -d -t passage-example.XXXXXXXX)"
cp -R examples/candidate-package "$PASSAGE_WORK/package"
passage-eval candidate freeze "$PASSAGE_WORK/package/source.json" \
  --out "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate validate "$PASSAGE_WORK/package/candidate.json"
passage-eval candidate register "$PASSAGE_WORK/package/candidate.json" \
  --catalog "$PASSAGE_WORK/catalog.json"
```

Expect a validated manifest and a catalog entry. The example is an unweighted
breadth-first search, with no learned weights and no benchmark result.

To run a package on real scenes, export an input-only cohort as in
[Submit a candidate](benchmark/candidate-registration.md#step-2-check-it-locally),
then run the worker directly with a new output directory:

```bash
python -m passage_eval.campaign.candidate.worker \
  --manifest "$PASSAGE_WORK/package/candidate.json" \
  --inputs "$PASSAGE_INPUTS" --output "$PASSAGE_WORK/run-cpu" --device cpu
```

The direct worker writes `journal.jsonl` and `run.json`. It is a quick local
check; a measured session uses the [rootless OCI runner](benchmark/candidate-oci.md),
which adds the container receipt, followed by host scoring. CUDA execution needs
compatible drivers, device exposure and an explicitly allowlisted GPU.

## :material-check-all: Run the focused checks

With the development and documentation dependencies installed, these commands
cover the candidate interface, the example and the documentation:

```bash
uv run --frozen pytest tests/candidates tests/passage_eval
uv run --frozen python -m unittest discover -s examples/candidate-package -p 'test_*.py'
uv run --frozen ruff check src/passage-eval/campaign/candidate tests/candidates
uv run --frozen --group docs mkdocs build --strict --site-dir "$PASSAGE_DIST/site"
```
