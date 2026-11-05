# PASSAGE edge base images: reconstruction notes

| Author | Date | Version |
| --- | --- | --- |
| Geoffrey Delhomme | 2026-11-04 | 0.2 — public source notes |

## 1. Scope and status

PASSAGE ships text recipes, package locks and context verification tools for
local builds. The release omits prebuilt images, SDK binaries, wheel archives,
Debian packages, candidate engines, weights and historical build contexts.
Thor and Orin require an explicit NVIDIA runtime selection. Tiger requires an
explicit OpenVINO runtime selection. The operator obtains those dependencies
from their providers under the applicable terms. An image build alone does not
qualify the persistent 256 evaluation or establish a leaderboard result.

**A complete public rebuild is not currently available.** The context builder also needs a separate inference source tree, which is not public at this time. The operator must obtain permission to access and use that source, pin an approved revision and review it. Publication and rights for that source remain open. The files here do not authorize distribution of that source, an SDK, an image or candidate weights.

## 2. Runtime inputs and historical identities

| Target | Platform | Expected native runtime | Historical context manifest SHA-256 |
| --- | --- | --- | --- |
| Thor | linux/arm64 | TensorRT 10.16.2.10, CUDA 13.2 | `c4c46080adcc12e0a250b1867073a706422f898800c073dae7e538e9267895d9` |
| Orin | linux/arm64 | TensorRT 10.16.2.10, CUDA 13.2 | `754899b3ecd5cdfe733e1e4f2e9caf08f874b076cdbc113b8aadb345d20a3af6` |
| Tiger | linux/amd64 | OpenVINO 2026.4.1 | `eca5f32251f10ebf6854c328bb2f3d0b9e162f33f5925e62c3896051a59aec22` |

These hashes identify historical manifests; the manifests and their contexts are not included in the public distribution. The current recipe has different source bytes and any new context and image require their own identities. The historical Tiger vendor image was `openvino/ubuntu24_dev@sha256:7512312434bd33892ce04b9bda86ffa94805fb35b7673b44b4f0cfe29917a91a`. The historical Thor and Orin SDK base was locally built, not a published NVIDIA image. A local image ID is not a registry manifest digest.

The profile-specific locks in `locks/` name Python packages and hashes; they are not the wheels or the target drivers. The SDKs, drivers, compatible base images and offline wheelhouses must be obtained separately under applicable terms. Review their provenance, licences and target compatibility before building.

## 3. Jetson SDK base

The text recipe and package inventory in `jetson-sdk/` describe an offline ARM64 SDK base. The package archives and signed repository indexes are not supplied. The historical NVIDIA signing-key fingerprint was not independently cross-checked; independently verify vendor keys, signatures, package identities and redistribution rights for the intended build. Dated repositories may no longer be available.

For a build, assemble a separately reviewed context containing `Dockerfile.sdk-base`, `verify_repo.py`, `packages.lock.json`, and `repo/` with the locked Debian archives in `repo/ubuntu` and `repo/jetson`. Copy `repo-SHA256SUMS` to `repo/SHA256SUMS`. The recipe verifies the pinned inventory and checks the package hashes before installation. The historical Ubuntu image reference was `docker.io/library/ubuntu@sha256:19cad59657037cffa0ac9a48107019b768900ff1071140965827513ac866bd5e`, and the package inventory SHA-256 is `49488c34dd541459c0b3a434451ba465a3871820bccf423ea4d2c77da7581f39`. These identify inputs, not a published SDK image or a proven reproduction.

Run a build only after independently reviewing the context and resolving the pinned Ubuntu base locally. With `SDK_CONTEXT` set to the reviewed context directory, a local build can be requested with:

```sh
UBUNTU_IMAGE='docker.io/library/ubuntu@sha256:19cad59657037cffa0ac9a48107019b768900ff1071140965827513ac866bd5e'
LOCK_SHA256=49488c34dd541459c0b3a434451ba465a3871820bccf423ea4d2c77da7581f39
docker image inspect "$UBUNTU_IMAGE"
docker build --pull=false --network=none --platform linux/arm64 \
  --build-arg "UBUNTU_IMAGE=$UBUNTU_IMAGE" --build-arg "LOCK_SHA256=$LOCK_SHA256" \
  --build-arg 'UBUNTU_SNAPSHOT=https://snapshot.ubuntu.com/ubuntu/20261025T160000Z' \
  -f "$SDK_CONTEXT/Dockerfile.sdk-base" -t local/passage-edge-sdk:reviewed "$SDK_CONTEXT"
```

`--network=none` restricts build steps; the builder may still resolve `FROM` metadata. Compatible target NVIDIA driver libraries must be provided separately at execution time. The SDK image has not been demonstrated to reproduce the historical image bit for bit.

## 4. Common edge context

The tools in `base/` expect sibling `passage` and `inference` source checkouts. The builder also requires a digest-pinned base image, a matching profile-specific lock and an independently acquired offline wheelhouse. The inference checkout is currently unavailable as public source, so the public files alone cannot complete this step.

Once the missing source and dependencies are lawfully available, inspect and pin both source revisions. Place the checkouts as sibling directories named `passage` and `inference`; run the builder from the PASSAGE checkout. Set `PROFILE`, `BASE_IMAGE`, `WHEELHOUSE`, `LOCK` and `CONTEXT` to the reviewed profile, immutable base reference, matching offline wheelhouse, matching lock and new context path outside the checkouts:

```sh
python3 containers/base/build_context.py --profile "$PROFILE" --base-image "$BASE_IMAGE" \
  --wheelhouse "$WHEELHOUSE" --lock "$LOCK" --context "$CONTEXT"
```

Record the printed manifest hash independently. Review all copied files and verify the context with `python3 containers/base/verify_context.py "$CONTEXT" --expected-manifest-sha256 "$MANIFEST_SHA256"` before building. The manifest binds context file bytes; it does not prove that an image was executed or that its output is correct. The Dockerfile leaves the default user as root: an operator must select a non-root UID and GID and independently establish device access, isolation and lease controls.

A local tag, image ID or configuration digest is not a registry manifest digest. If a digest-addressed image is needed, obtain the actual registry manifest digest from an authorized registry, verify its bytes, platform and configuration, and use that reference. Do not publish SDK components or weights until their distribution rights are resolved.

## 5. Target execution and qualification

An experimental Tiger root-filesystem launch ran Seg and Reg 2026 on 128 previously exposed validation scenes with OpenVINO reporting GPU.0. This is feasibility evidence for that particular environment. It does not establish a publicly rebuildable image, qualification, or an equivalent route on Thor or Orin. Its OCI-layer-to-rootfs binding was not independently established for that experiment.

Before an admitted run, the operator must establish the applicable source and image identities, target-specific execution and device placement, and the required admission and validation evidence. These draft recipes confer no admission or `verified` status. See [LICENSING.md](LICENSING.md) for the unresolved rights review.
