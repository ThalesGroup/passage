# Run a candidate with rootless OCI

Use this runner to execute an image-installed PASSAGE worker when the host Docker
runtime cannot create its systemd cgroup scope. Docker supplies an export of an
already-local image; `runc --rootless true` executes the exported filesystem with
private runtime state. No daemon configuration changes are required.

The runner records filesystem and process evidence. It does not implement strict
benchmark admission, resource cgroups, or reconstruction of the image from its
layers. See [Contributing a method](contributing-a-method.md) for the surrounding
candidate workflow.

## Prepare an image once

The host needs Docker access for inspection and export, a working rootless runc,
and permission to create user, mount, PID, network, IPC, UTS and cgroup namespaces.
The image must contain Python and
`passage_eval.campaign.candidate.worker`. It must support a single mapped UID/GID.
The existing host policy must permit rootless namespaces; the runner does not
change AppArmor, systemd, sysctls, setuid binaries or GPU configuration.

Choose a fresh destination on a filesystem with space for both the export archive
and its extracted root. Preparation retains both as evidence.

```bash
python -m passage_eval.campaign.candidate.oci prepare \
  --image local/passage-seg-candidate:cpu-arm64-20261111 \
  --output /absolute/path/to/oci-runtime/seg-rootfs
```

The result is `seg-rootfs/rootfs-receipt.json`. It identifies the local image using
its observed Docker inspect ID, descriptor and repository digests. Docker inspect
may expose an OCI index ID rather than a config digest. The receipt therefore
keeps config metadata separate, reports unavailable config or platform-manifest
digests as null, and hashes the actual export archive and extracted tree.
Extraction preserves container symlinks without following them into the host;
setuid/setgid bits are removed and ownership is materialized as the calling user.

Preparation uses a unique `passage-pilot-export-*` container and removes that
container after export, including on export failure. It does not start it or pull
an image. An existing output directory is rejected.

## Run a CPU session

Provide a sealed candidate manifest and **one input-only NPZ file**. The archive
must contain exactly `sample_ids`, `terrain`, `obstacles`, `starts` and `goals`;
reference costs, labels, indices and prepared training archives are rejected.
The worker validates the array shapes, types and declared scene contract before
importing candidate code.

```bash
python -m passage_eval.campaign.candidate.oci run \
  --rootfs /absolute/path/to/oci-runtime/seg-rootfs/rootfs-receipt.json \
  --manifest /path/to/package/candidate.source.json \
  --inputs /path/to/inputs/scenes.npz \
  --output /absolute/path/to/oci-runtime/seg-run \
  --device cpu \
  --timeout-seconds 300
```

The output directory must be new and disjoint from the package and prepared root.
The manifest may have any filename. Only its declared artifacts are copied into a
fresh execution package with the manifest named `candidate.json`. This snapshot
and the runtime configuration live outside the writable output mount.

The container receives these host binds:

| Container path | Host source | Access |
|---|---|---|
| `/candidate` | Snapshot containing declared package artifacts and manifest | Read only |
| `/inputs/scenes.npz` | The single NPZ file | Read only |
| `/output` | The new output directory | Read/write |

The input file's parent directory is never bound. The root is read only; `/tmp`,
`/dev` and shared memory are private mounts. The process has no Linux capabilities
and has `noNewPrivileges=true`. Network, PID, user, mount, IPC, UTS and cgroup
namespaces are separate from the host. CPU execution receives no GPU device binds.
The worker runs as namespace UID/GID 0, mapped to the calling host user and group.

The command prints a JSON receipt and writes `oci-run.json` alongside the worker's
`run.json` and `journal.jsonl`. It records the image, manifest and input identities,
rootfs tree hashes before and after execution, the private runtime location,
process return code, timeout and cleanup results. The archive and prepared root
are checked before launch; the root, staged package, original package and input
are checked again as applicable after execution.

A nonzero worker exit or runtime timeout gives a nonzero CLI exit status. Python
callers receive a receipt containing that status; they must check `returncode` and
`timed_out` before treating the session as successful. Preflight or integrity
failures raise an exception and retain diagnostics in the private runtime folder
when it has been created.

## Understand timeout and cleanup evidence

`--timeout-seconds` bounds the runc execution interval, including worker startup.
Image preparation, staging and hashing occur outside that interval. On a timeout,
the host asks runc to kill all processes in its private instance, kills and reaps
the runtime process group, and force-deletes the instance. Kill, force-delete and
state/PID inspection are attempted independently even when
one command raises an exception or times out. Each error is accumulated in the
receipt; an earlier execution exception is preserved in `failure.json`.
The receipt checks that the private runtime state is empty and the recorded init
PID is absent. If an inspection fails, its state and `child_namespace_drained`
are null (unknown), rather than true. Cleanup exceptions cause the API to fail
after retaining the available receipt and diagnostics.

`child_namespace_drained` concerns process extinction: it is derived from the
private runc state and init PID, rather than an observation of GPU or DMA
completion. A successful cooperative worker `synchronize` hook and a forced
SIGKILL are different events. No hardware drain or reset after a timeout is
established by this launcher.

The cgroup namespace changes visibility; it does not enforce memory or CPU limits.
Every receipt sets `resource_cgroup_enforced=false`. The runner does not add a
seccomp filter. A timed-out process can leave partial worker output, which must
not be scored as a completed run. Keep the private runtime directory for its
configuration, stdout/stderr, rootfs manifest and failure evidence.

## Inspect the optional CUDA plan

CUDA support is restricted to the explicitly assigned UUID
`GPU-1ae3b771-4e59-bf00-5c9d-06e82bd84ece`. Planning reads the CDI file and stats
character devices without opening a GPU or querying CUDA:

```bash
python -m passage_eval.campaign.candidate.oci cdi-plan \
  --gpu-uuid GPU-1ae3b771-4e59-bf00-5c9d-06e82bd84ece \
  --output /path/to/cdi-plan.json
```

The plan combines only this UUID's edits with `common containerEdits`; aliases for
other GPUs and `all` are ignored. It validates character-device major/minor
numbers and records the CDI hash, library/file hashes, resolved symlinks and
socket metadata. The YAML is parsed and hashed from the same byte sequence.
Before launch, each regular dependency is copied into a private runtime snapshot
outside `/output`, and its copied bytes must match the planned hash. Versioned
library mounts and all aliases use these copies. Their hashes and modes are
checked immediately before and after execution. Replacing a host library after
snapshotting does not change the bytes bound into this session. A mismatch or a
mutation of the private snapshot fails the run explicitly. This assumes other
processes of the operator's UID do not tamper with the private runtime during
execution.

Character devices are checked again by type, major/minor and mode. Sockets are
checked by resolved path, type, mode and inode; no content hash is claimed for
either. Runtime device access uses individual bind mounts rather than
privileged device-node creation. Library aliases become read-only file binds;
`LD_LIBRARY_PATH` supplies the driver library folders. Original CDI hooks remain
in the receipt but are never executed on the host. In particular, this runner
does not perform NVIDIA's CUDA compatibility hook or claim full hook equivalence.
The host CDI specification remains a trust boundary. Common NVIDIA control/UVM
nodes and the persistence socket can expose interfaces beyond an individual
device node. Their inclusion does not prove that a candidate cannot enumerate
other GPUs. A read-only bind of a Unix socket does not establish that the socket
protocol has no side effects. Neither this plan nor its UUID selection establishes
hardware or DMA confinement without additional evidence.

After the operator has assigned a GPU budget, a CUDA-capable candidate can use the
same `run` command with `--device cuda --gpu-uuid` and the UUID above. A passive
CDI plan is not evidence of a successful CUDA run, driver compatibility or device
synchronization. Those require a separately recorded execution and worker drain.

## Python API

```python
from pathlib import Path
from passage_eval.campaign.candidate.oci import prepare_rootfs, run_oci

prepare_rootfs("local/passage-seg-candidate:cpu-arm64-20261111", Path("/data/rootfs"))
receipt = run_oci(
    Path("/data/rootfs/rootfs-receipt.json"),
    Path("/data/package/candidate.json"),
    Path("/data/inputs/scenes.npz"),
    Path("/data/session"),
    device="cpu",
    timeout_seconds=300,
)
assert receipt["returncode"] == 0 and not receipt["timed_out"]

from passage_eval.campaign.candidate.binding import bind_container_evidence
bind_container_evidence("/data/session")
```

Both functions return JSON-serializable dictionaries. Large rootfs archives and
per-run trees belong on a data volume. Review this procedure when the image,
runc version, host namespace policy, CDI specification or allowed GPU changes.
The campaign operator owns execution scheduling and resource budgets.
