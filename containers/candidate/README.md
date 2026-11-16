# Build the private ML candidate runtime

This ARM64 CPython 3.12 image supplies the common PASSAGE worker, Torch with CPU and CUDA 13.0 code, Stable Baselines3, SB3-Contrib and Gymnasium. Candidate code and weights belong to the separately mounted `/candidate`; graph and reinforcement-learning source are not baked into the common image.

The private build kit captures an existing installed environment and a separately built core wheel. Its file hashes, exact distribution inventory and pinned Python base define the inputs. It does not claim to reproduce a historical native image or resolve the training repositories' dependency locks.

## Prerequisites

This recipe needs authorized access to the existing ARM64 CPython 3.12 runtime
environment, its installed dependencies, a compatible image builder and the
pinned base image. Those inputs are not distributed by this guide. There is no
public CUDA runtime image or package-index availability claim. For the CPU
example and exact core installation, start with the
[release guide](../../docs/release.md) and
[candidate operator guide](../../docs/benchmark/candidate-registration.md).

## Prepare a verified kit

Use the approved CPU/CUDA runtime environment and a **freshly built official
1.0.0 candidate wheel** containing `passage_eval.campaign.candidate.worker`.
Build it from the reviewed source and verify its SHA-256 using the release guide.
The filename alone is insufficient: historical private wheels also used 1.0.0,
and private measurement/runtime artifacts have also used 1.1.0. Preserve those
files and receipts; never rename or reuse them as the new release. This command
reads the source environment and creates a new context:

```sh
python3.12 containers/candidate/prepare_ml_context.py \
  --runtime-env /absolute/path/to/runtime-venv \
  --core-wheel /absolute/path/to/core-wheels/passage-1.0.0-py3-none-any.whl \
  --output /absolute/path/to/new-ml-image-context
```

Place the context on the same filesystem as the source environment to use hardlinks. Files are never rewritten through those links, and only the editable PASSAGE `.pth` and finder files are omitted from the context. The source environment keeps those files. A cross-filesystem context uses copies instead. Existing output directories and source symlinks are rejected.

Keep the whole kit for reconstruction: `site-packages/`, `core-wheels/`, `snapshot.files.json`, `runtime.snapshot.lock.json`, the Containerfile and verification scripts. The observed requirements text is an inventory, not an install command. The JSON lock binds each snapshot file through the SHA-256 manifest and binds the supplied core wheel. Verify it before reusing a kit, since a hardlink is not a write-protection mechanism.

The context metadata, wheel, recipe and verifier hashes can be checked directly:

```sh
cd /absolute/path/to/new-ml-image-context
sha256sum -c context.SHA256SUMS
```

The image build separately verifies every installed snapshot file against `snapshot.files.json` before replacing the editable core with the supplied wheel.

## Build without a resolver

```sh
docker build --network=none \
  -f /absolute/path/to/new-ml-image-context/ML.Containerfile \
  -t local/passage-candidate-ml:1.0.0-preparation \
  /absolute/path/to/new-ml-image-context
```

The base is `python:3.12@sha256:17b9724493d916f56565c131f389faff2597f46dec6ce09516f95125bc9f519f`. It must already be available locally, or be acquired before the offline build. The build starts with an empty image-side site-packages directory, copies the exact snapshot, verifies its full file set and hashes, bootstraps pip offline from the pinned Python base and installs the supplied core wheel with `--no-index --no-deps --force-reinstall`.

Before the CPU probe, a metadata guard reads the supplied wheel's sole
`.dist-info/METADATA` from its ZIP archive and checks that the installed core's
name, version, `Requires-Python` and `Requires-Dist` match it. This also detects
stale installed requirements when historical wheels share the same version.
It checks the wheel's Python requirement and active default dependencies
against the image's Python and installed versions. Markers use that target
environment with no core extras selected. The guard uses `packaging`, already
required by the approved Torch runtime; it does not import ML packages,
resolve dependencies, or inspect unrelated NVIDIA wheel tags. Active direct
URL requirements fail because this check cannot establish their installed
origin.

The approved snapshot must already satisfy the new wheel's requirements:
`--no-deps` does not upgrade it. For example, historical Loguru 0.7.3 and ImageIO
2.38.1 do not satisfy the fresh wheel's `loguru>=0.8.0` and `imageio>=2.38.2`.
On a mismatch, prepare an approved fresh runtime prefix and a new kit; preserve
the historical environment, kit, inventories and receipts. A historical kit
rebuilds only with its original recorded wheel.

A CPU-only build probe imports Torch, SB3-Contrib, Gymnasium and the actual core worker, and performs a tiny CPU matrix multiplication. It records the CUDA build version without querying or initializing a CUDA device. The final environment leaves CUDA visibility under the execution launcher's control. GPU execution still needs compatible driver/device exposure and its own observed test.

## Use the common worker

Mount one materialized candidate package at `/candidate`, an input-only NPZ at `/inputs/scenes.npz`, and a writable directory at `/output`. The default entry point is:

```sh
python -m passage_eval.campaign.candidate.worker \
  --manifest /candidate/candidate.json \
  --inputs /inputs/scenes.npz --output /output --device cpu
```

A CUDA launch changes the device argument and explicitly supplies the approved device/driver exposure. Keep the candidate and input mounts read-only and the measurement network disabled. The host owns scoring references, time supervision and admission; the image returns ordinary candidate outputs.

The image fixes CPU numerical-library thread limits at one. The launcher owns any process-wide CPU quota. Mutable model outputs, Python caches and plotting configuration use writable `/tmp` or `/output`; candidate source can remain read-only.

## Interpret the recorded versions

The [observed runtime lock](runtime-observed-20261111.lock.json) is an advisory redacted copy of a historical
receipt for its private kit, including the wheel hash recorded there. Its local
source path was replaced in this public projection; its bytes do not preserve
the original receipt identity recorded in the projection manifest. It is not
the lock for a newly prepared 1.0.0 release runtime. A new kit must generate its
own lock and receipts using the freshly built wheel; retain the historical
1.1.0 runtime identities and any older same-version 1.0.0 wheel hashes unchanged.

The 2026-11-11 snapshot observed NumPy 2.4.6, Numba 0.65.1, llvmlite 0.47.0, Torch metadata 2.13.0 with build version 2.13.0+cu130, SB3-Contrib/Stable Baselines3 2.9.2 and Gymnasium 1.4.0. An earlier shared-runtime observation reported NumPy 2.4.4. This kit records the actual current files and does not claim that older observation or a repository `uv.lock` was executed unchanged.

The kit is ARM64 and CPython 3.12 specific. Its installed-file lock is different from a wheel-resolution lock: reconstruct from the retained private kit and pinned base, rather than installing the inventory from a package index. The build probe and final launch records distinguish CPU behavior, compiled CUDA support and CUDA execution.

## Verify the builder

```sh
python3.12 -m unittest discover -s containers/candidate -p 'test_*.py'
```

The tests verify source preservation, selective editable-file removal, hardlink
use, unexpected-file detection and symlink rejection. Real temporary metadata
also exercises the core requirement guard, including outdated or missing
dependencies, default markers, Python compatibility and legacy NVIDIA tags.
Final image evidence must record the local configuration digest, platform
manifest digest and OCI index digest separately, with their observed media
types. No upload is part of these commands.
