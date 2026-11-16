"""Operator-local rootless OCI execution; resource cgroups and strict admission are absent.

Docker is used only to inspect/create/export an existing local image. Runtime state,
logs and mounts belong to this run; no daemon or system configuration is changed.
"""

from __future__ import annotations

import argparse
import errno
from collections import deque
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time
from typing import Any
from uuid import uuid4
import zipfile

from passage_eval.candidates.catalog import atomic_json
from passage_eval.candidates.staging import stage_candidate_package
from passage_eval.protocol.candidate_package import file_sha256, load_candidate_package, read_json

ALLOWED_GPU = "GPU-1ae3b771-4e59-bf00-5c9d-06e82bd84ece"
INPUT_KEYS = {"sample_ids.npy", "terrain.npy", "obstacles.npy", "starts.npy", "goals.npy"}
ROOTFS_SCHEMA = "passage-rootless-rootfs/v1"


def tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash paths, modes, file contents and link targets, never traversing links."""
    records = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            info = path.lstat()
            item = {"path": path.relative_to(root).as_posix(), "mode": stat.S_IMODE(info.st_mode)}
            if stat.S_ISLNK(info.st_mode):
                item.update(type="symlink", target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                item.update(type="directory")
            elif stat.S_ISREG(info.st_mode):
                item.update(type="file", size=info.st_size, sha256=file_sha256(path))
            else:
                raise ValueError(f"rootfs contains unsupported special file: {path}")
            records.append(item)
    records.sort(key=lambda item: item["path"])
    data = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records)
    return sha256(data.encode()).hexdigest(), records


def _copy_root_file(source: str | Path, target: str | Path) -> str | Path:
    """Share immutable file bytes locally; copy when output uses another volume."""
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
    return target


def _archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"archive path escapes rootfs: {name}")
    return path


def _extract_rootfs(archive: Path, root: Path) -> None:
    """Defer links until file extraction; a tar member can never traverse a link.

    Absolute image symlinks are retained as container paths, not followed by the
    host. Image ownership is materialized as the calling host user for one-ID maps.
    """
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        links = {_archive_path(m.name).as_posix() for m in members if m.issym() or m.islnk()}
        for member in members:
            relative = _archive_path(member.name)
            if relative.as_posix() == ".":
                continue
            if any(parent.as_posix() in links for parent in relative.parents):
                raise ValueError(f"archive member has a symlink ancestor: {member.name}")
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError(f"unsupported archive member: {member.name}")
        # Directories remain writable until extraction is complete.
        for member in members:
            relative = _archive_path(member.name)
            if relative.as_posix() == ".":
                continue
            path = root.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f"missing archive bytes: {member.name}")
                with source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)  # never copy setuid/setgid bits
        for member in members:
            path = root.joinpath(*_archive_path(member.name).parts)
            if member.islnk():
                target = root.joinpath(*_archive_path(member.linkname).parts)
                if not target.is_file() or target.is_symlink():
                    raise ValueError("archive hardlink must identify an extracted regular file")
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, path)
        for member in members:
            if member.issym():
                path = root.joinpath(*_archive_path(member.name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(member.linkname)
        for member in reversed(members):
            if member.isdir() and _archive_path(member.name).as_posix() != ".":
                root.joinpath(*_archive_path(member.name).parts).chmod(member.mode & 0o777)


def _container_path(root: Path, destination: str) -> Path:
    """Resolve image symlinks as container paths, including absolute link targets."""
    if not destination.startswith("/") or ".." in PurePosixPath(destination).parts:
        raise ValueError("mount destination must be an absolute container path")
    pending = deque(PurePosixPath(destination).parts[1:])
    parts: list[str] = []
    links = 0
    while pending:
        name = pending.popleft()
        if name in ("", "."):
            continue
        if name == "..":
            if parts:
                parts.pop()
            continue
        path = root.joinpath(*parts, name)
        if path.is_symlink():
            links += 1
            if links > 40:
                raise ValueError("container symlink cycle")
            target = PurePosixPath(os.readlink(path))
            if target.is_absolute():
                parts = []
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            parts.append(name)
    return root.joinpath(*parts)


def _mountpoint(root: Path, destination: str, *, directory: bool) -> None:
    path = _container_path(root, destination)
    if directory:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        if not path.is_file():
            raise ValueError(f"mount point is not a file: {destination}")


def prepare_rootfs(image: str, out: Path) -> dict[str, Any]:
    """Export one already-local image into a fresh operator-owned rootfs receipt."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inspect = json.loads(
        subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True, text=True).stdout
    )[0]
    atomic_json(out / "image-inspect.json", inspect)
    name = "passage-pilot-export-" + uuid4().hex
    archive = out / "rootfs.tar"
    created = False
    try:
        subprocess.run(
            ["docker", "create", "--name", name, "--pull=never", inspect["Id"], "/bin/true"],
            check=True,
            capture_output=True,
            text=True,
        )
        created = True
        subprocess.run(["docker", "export", name, "--output", str(archive)], check=True, capture_output=True, text=True)
    finally:
        if created:
            subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True, text=True)
    root = out / "rootfs"
    _extract_rootfs(archive, root)
    digest, records = tree_manifest(root)
    atomic_json(out / "rootfs-tree.json", {"sha256": digest, "entries": records})
    descriptor = inspect.get("Descriptor")
    media = (descriptor or {}).get("mediaType", "")
    config = inspect.get("Config", {})
    receipt = {
        "schema": ROOTFS_SCHEMA,
        "image_requested": image,
        "image_inspect_id": inspect["Id"],
        "image_descriptor": descriptor,
        "observed_repo_digests": inspect.get("RepoDigests", []),
        "platform_manifest_digest": (descriptor or {}).get("digest") if media.endswith("manifest.v1+json") else None,
        "image_config_digest": (descriptor or {}).get("digest") if media.endswith("config.v1+json") else None,
        "image_config_metadata": config,
        "image_config_metadata_sha256": sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "architecture": inspect.get("Architecture"),
        "os": inspect.get("Os"),
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
        "rootfs": str(root),
        "rootfs_tree_sha256": digest,
        "rootfs_entries": len(records),
        "tree_manifest": str(out / "rootfs-tree.json"),
        "image_inspect_file_sha256": file_sha256(out / "image-inspect.json"),
        "ownership": "single host UID/GID; archive setuid/setgid bits removed",
        "reproduced_from_layers": False,
        "strict_admission": False,
    }
    atomic_json(out / "rootfs-receipt.json", receipt)
    return receipt


def _check_inputs(path: Path) -> None:
    if not path.is_file():
        raise ValueError("inputs must be one input-only NPZ file")
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if (
            len(items) != len(INPUT_KEYS)
            or {item.filename for item in items} != INPUT_KEYS
            or sum(item.file_size for item in items) > 2 * 1024**3
        ):
            raise ValueError("inputs must contain only the five input-only arrays, at most 2 GiB")


def _dependency(path: Path) -> dict[str, Any]:
    path = Path(path)
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    item = {
        "path": str(path),
        "resolved": str(resolved),
        "mode": stat.S_IMODE(info.st_mode),
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
    }
    if stat.S_ISREG(info.st_mode):
        item.update(type="file", sha256=file_sha256(resolved))
    elif stat.S_ISSOCK(info.st_mode):
        item.update(type="socket", sha256=None, inode=info.st_ino)
    else:
        raise ValueError(f"CDI mount must be a file or socket: {path}")
    return item


def prepare_cdi(gpu_uuid: str, path: Path = Path("/var/run/cdi/nvidia.yaml")) -> dict[str, Any]:
    """Passively read the selected GPU and common CDI edits; never open a GPU."""
    if gpu_uuid != ALLOWED_GPU:
        raise ValueError("only the explicitly approved GPU UUID is supported")
    import yaml

    cdi_bytes = Path(path).read_bytes()
    document = yaml.safe_load(cdi_bytes)
    if document.get("kind") != "nvidia.com/gpu":
        raise ValueError("expected NVIDIA CDI specification")
    matches = [item for item in document["devices"] if item["name"] == gpu_uuid]
    if len(matches) != 1:
        raise ValueError("approved GPU must occur exactly once in CDI")
    edits = [document.get("containerEdits", {}), matches[0].get("containerEdits", {})]
    mounts, dependencies, nodes, symlinks, hooks, env = [], [], [], [], [], []
    for edit in edits:
        for mount in edit.get("mounts", []):
            dependency = _dependency(Path(mount["hostPath"]))
            dependencies.append(dependency)
            destination = mount["containerPath"]
            if not destination.startswith("/") or ".." in PurePosixPath(destination).parts:
                raise ValueError("invalid CDI destination")
            mounts.append(
                {
                    "hostPath": dependency["resolved"],
                    "containerPath": destination,
                    "options": sorted(set(mount.get("options", [])) | {"ro", "nosuid", "nodev"}),
                }
            )
        for node in edit.get("deviceNodes", []):
            host = Path(node.get("hostPath", node["path"])).resolve(strict=True)
            info = host.stat()
            if not stat.S_ISCHR(info.st_mode):
                raise ValueError(f"CDI requires a character device: {host}")
            if (os.major(info.st_rdev), os.minor(info.st_rdev)) != (node.get("major", 0), node.get("minor", 0)):
                raise ValueError(f"CDI character device numbers differ: {host}")
            nodes.append(
                {
                    "path": node["path"],
                    "hostPath": str(host),
                    "major": os.major(info.st_rdev),
                    "minor": os.minor(info.st_rdev),
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        env.extend(edit.get("env", []))
        for hook in edit.get("hooks", []):
            hooks.append(hook)
            args = hook.get("args", [])
            if len(args) > 1 and args[1] == "create-symlinks":
                for index, argument in enumerate(args):
                    if argument == "--link":
                        target, link = args[index + 1].split("::", 1)
                        if not link.startswith("/") or ".." in PurePosixPath(link).parts:
                            raise ValueError("invalid CDI symlink destination")
                        symlinks.append({"link": link, "target": target})
    return {
        "schema": "passage-cdi-plan/v1",
        "gpu_uuid": gpu_uuid,
        "cdi_path": str(path),
        "cdi_sha256": sha256(cdi_bytes).hexdigest(),
        "mounts": mounts,
        "device_nodes": nodes,
        "dependencies": dependencies,
        "symlinks": symlinks,
        "env": env,
        "original_hooks": hooks,
        "hooks_executed": False,
        "strategy": "read-only alias binds and LD_LIBRARY_PATH; no host hooks",
        "cuda_compat_hook_executed": False,
        "gpu_execution_validated": False,
    }


def _snapshot_cdi(plan: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Copy, never hardlink, regular CDI dependencies outside writable output.

    The bytes copied must match the passive plan. All versioned library mounts
    and their aliases then refer to the independently verified snapshot.
    """
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(json.dumps(plan))
    copied = {}
    for index, dependency in enumerate(snapshot["dependencies"]):
        if dependency["type"] != "file":
            continue
        source = Path(dependency["resolved"])
        target = destination / f"{index:03d}-{dependency['sha256']}"
        shutil.copyfile(source, target)
        if file_sha256(target) != dependency["sha256"]:
            raise ValueError(f"CDI dependency changed before snapshot: {source}")
        target.chmod(dependency["mode"] & 0o777)
        dependency["snapshot_path"] = str(target)
        copied[str(source)] = str(target)
    for mount in snapshot["mounts"]:
        mount["hostPath"] = copied.get(mount["hostPath"], mount["hostPath"])
    snapshot["snapshot_directory"] = str(destination)
    snapshot["regular_file_strategy"] = "independent copies verified before and after execution"
    _verify_cdi_snapshot(snapshot)
    return snapshot


def _verify_cdi_snapshot(plan: dict[str, Any]) -> dict[str, int]:
    counts = {"regular_files": 0, "character_devices": 0, "sockets": 0}
    for dependency in plan["dependencies"]:
        if dependency["type"] == "file":
            path = Path(dependency["snapshot_path"])
            if (
                not path.is_file()
                or path.is_symlink()
                or file_sha256(path) != dependency["sha256"]
                or stat.S_IMODE(path.stat().st_mode) != (dependency["mode"] & 0o777)
            ):
                raise ValueError(f"CDI snapshot changed: {path}")
            counts["regular_files"] += 1
        elif dependency["type"] == "socket":
            current = _dependency(Path(dependency["path"]))
            if current != dependency:
                raise ValueError(f"CDI socket identity changed: {dependency['path']}")
            counts["sockets"] += 1
        else:
            raise ValueError("unsupported CDI dependency type")
    for node in plan["device_nodes"]:
        info = Path(node["hostPath"]).stat()
        if (
            not stat.S_ISCHR(info.st_mode)
            or (os.major(info.st_rdev), os.minor(info.st_rdev)) != (node["major"], node["minor"])
            or stat.S_IMODE(info.st_mode) != node["mode"]
        ):
            raise ValueError(f"CDI character device identity changed: {node['hostPath']}")
        counts["character_devices"] += 1
    return counts


def _bind(source: Path | str, destination: str, *, readonly: bool = True) -> dict[str, Any]:
    return {
        "destination": destination,
        "type": "bind",
        "source": str(source),
        "options": ["bind", "ro" if readonly else "rw", "nosuid", "nodev", "private"],
    }


def _build_spec(
    root: Path, package: Path, inputs: Path, output: Path, *, python: str, env: list[str], device: str
) -> dict[str, Any]:
    environment = dict(item.split("=", 1) for item in env if "=" in item)
    environment.update(
        HOME="/tmp",
        PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR="/tmp/numba-cache",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    if device == "cpu":
        environment.update(CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void")
    return {
        "ociVersion": "1.2.1",
        "hostname": "passage-candidate",
        "root": {"path": str(root), "readonly": True},
        "process": {
            "terminal": False,
            "user": {"uid": 0, "gid": 0},
            "args": [
                python,
                "-m",
                "passage_eval.campaign.candidate.worker",
                "--manifest",
                "/candidate/candidate.json",
                "--inputs",
                "/inputs/scenes.npz",
                "--output",
                "/output",
                "--device",
                device,
            ],
            "env": [f"{key}={value}" for key, value in sorted(environment.items())],
            "cwd": "/candidate",
            "noNewPrivileges": True,
            "capabilities": {key: [] for key in ["bounding", "effective", "inheritable", "permitted", "ambient"]},
            "rlimits": [{"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024}],
        },
        "mounts": [
            {"destination": "/proc", "type": "proc", "source": "proc", "options": ["nosuid", "nodev", "noexec"]},
            {
                "destination": "/dev",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "strictatime", "mode=755", "size=65536k"],
            },
            {
                "destination": "/dev/pts",
                "type": "devpts",
                "source": "devpts",
                "options": ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"],
            },
            {
                "destination": "/dev/shm",
                "type": "tmpfs",
                "source": "shm",
                "options": ["nosuid", "nodev", "mode=1777", "size=65536k"],
            },
            {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs", "options": ["nosuid", "nodev", "mode=1777"]},
            _bind(package, "/candidate"),
            _bind(inputs, "/inputs/scenes.npz"),
            _bind(output, "/output", readonly=False),
        ],
        "linux": {
            "uidMappings": [{"containerID": 0, "hostID": os.getuid(), "size": 1}],
            "gidMappings": [{"containerID": 0, "hostID": os.getgid(), "size": 1}],
            "namespaces": [{"type": name} for name in ["pid", "ipc", "uts", "mount", "network", "cgroup", "user"]],
            "maskedPaths": [
                "/proc/acpi",
                "/proc/asound",
                "/proc/kcore",
                "/proc/keys",
                "/proc/latency_stats",
                "/proc/timer_list",
                "/proc/timer_stats",
                "/proc/sched_debug",
                "/proc/scsi",
            ],
            "readonlyPaths": ["/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger"],
        },
    }


def _apply_cdi(spec: dict[str, Any], plan: dict[str, Any]) -> None:
    bindings = {item["containerPath"]: item["hostPath"] for item in plan["mounts"]}
    bindings.update({item["path"]: item["hostPath"] for item in plan["device_nodes"]})
    pending = list(plan["symlinks"])
    while pending:
        remaining = []
        for item in pending:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(item["link"]), item["target"]))
            if target in bindings:
                bindings[item["link"]] = bindings[target]
            else:
                remaining.append(item)
        if len(remaining) == len(pending):
            raise ValueError(f"CDI symlink targets have no validated binding: {remaining}")
        pending = remaining
    nodes = {item["path"] for item in plan["device_nodes"]}
    for destination, source in bindings.items():
        mount = _bind(source, destination)
        if destination in nodes or stat.S_ISCHR(Path(source).stat().st_mode):
            mount["options"] = ["bind", "rw", "nosuid", "private"]  # char nodes must not use nodev
        spec["mounts"].append(mount)
    environment = dict(item.split("=", 1) for item in spec["process"]["env"])
    environment.update(item.split("=", 1) for item in plan["env"])
    folders = sorted(
        {posixpath.dirname(item["containerPath"]) for item in plan["mounts"] if ".so" in item["containerPath"]}
    )
    environment["LD_LIBRARY_PATH"] = ":".join(folders + [environment.get("LD_LIBRARY_PATH", "")])
    environment["CUDA_VISIBLE_DEVICES"] = plan["gpu_uuid"]
    spec["process"]["env"] = [f"{key}={value}" for key, value in sorted(environment.items())]


def _execute(args: list[str], stdout: Path, stderr: Path, *, timeout_seconds: float, on_timeout=None) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    started = time.monotonic()
    timed_out = False
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        try:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                if on_timeout:
                    on_timeout()
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "process_reaped": True,
        "elapsed_seconds": time.monotonic() - started,
    }


def _runc_action(
    command_base: list[str], arguments: list[str], action: str, cleanup: list[dict[str, Any]]
) -> subprocess.CompletedProcess | None:
    """A failed cleanup command is evidence, never a reason to skip the next one."""
    try:
        attempt = subprocess.run(command_base + arguments, capture_output=True, text=True, timeout=10)
    except Exception as error:
        cleanup.append({"action": action, "error_type": type(error).__name__, "error": str(error)})
        return None
    cleanup.append({"action": action, "returncode": attempt.returncode, "stderr": attempt.stderr})
    return attempt


def _cleanup_runtime(
    command_base: list[str], instance: str, runtime: Path, cleanup: list[dict[str, Any]]
) -> dict[str, Any]:
    _runc_action(command_base, ["kill", "--all", instance, "KILL"], "kill_all", cleanup)
    _runc_action(command_base, ["delete", "--force", instance], "delete_force", cleanup)
    attempt = _runc_action(command_base, ["list", "--format", "json"], "inspect_state", cleanup)
    states = None
    try:
        if attempt is None:
            raise RuntimeError("private runtime inspection did not complete")
        if attempt.returncode != 0:
            raise RuntimeError("private runtime inspection returned nonzero")
        states = json.loads(attempt.stdout)
        if states is None:  # runc 1.3 emits null for an empty instance list
            states = []
        if not isinstance(states, list):
            raise ValueError("private runtime state must be a JSON list")
    except Exception as error:
        states = None
        cleanup.append({"action": "decode_state", "error_type": type(error).__name__, "error": str(error)})
    init_pid = None
    init_absent = None
    try:
        pid_file = runtime / "init.pid"
        init_pid = int(pid_file.read_text()) if pid_file.exists() else None
        init_absent = init_pid is None or not Path(f"/proc/{init_pid}").exists()
    except Exception as error:
        cleanup.append({"action": "inspect_init", "error_type": type(error).__name__, "error": str(error)})
    state_empty = None if states is None else not states
    drained = None if state_empty is None or init_absent is None else state_empty and init_absent
    return {
        "private_runtime_state_empty": state_empty,
        "init_host_pid": init_pid,
        "init_process_absent": init_absent,
        "child_namespace_drained": drained,
        "cleanup_errors": [item for item in cleanup if "error_type" in item],
    }


def run_oci(
    rootfs_receipt: Path,
    package: Path,
    inputs: Path,
    output: Path,
    *,
    device: str = "cpu",
    gpu_uuid: str | None = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Execute the image-installed worker with only package, input file and output bound."""
    if device not in {"cpu", "cuda"} or (device == "cpu" and gpu_uuid is not None):
        raise ValueError("invalid device/GPU selection")
    if device == "cuda" and gpu_uuid != ALLOWED_GPU:
        raise ValueError("CUDA requires the explicitly approved GPU UUID")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    receipt_path = Path(rootfs_receipt).resolve()
    root_receipt = read_json(receipt_path)
    if root_receipt.get("schema") != ROOTFS_SCHEMA:
        raise ValueError("unknown rootfs receipt schema")
    base = Path(root_receipt["rootfs"]).resolve()
    if tree_manifest(base)[0] != root_receipt["rootfs_tree_sha256"]:
        raise ValueError("prepared rootfs tree changed")
    if file_sha256(Path(root_receipt["archive"])) != root_receipt["archive_sha256"]:
        raise ValueError("export archive changed")
    manifest = Path(package).resolve()
    if manifest.is_dir():
        manifest /= "candidate.json"
    candidate = load_candidate_package(manifest)
    inputs = Path(inputs).resolve()
    _check_inputs(inputs)
    input_hash = file_sha256(inputs)
    output = Path(output).resolve()
    for protected in (candidate.root, base):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output and protected directories must be disjoint")
    if inputs.is_relative_to(output):
        raise ValueError("output cannot contain the input file")
    if device not in candidate.document["backends"]:
        raise ValueError("device is not declared by candidate")
    plan = prepare_cdi(gpu_uuid) if device == "cuda" else None
    output.mkdir(parents=True, exist_ok=False)
    runtime = Path(tempfile.mkdtemp(prefix=".passage-oci-", dir=output.parent))
    instance = "passage-pilot-run-" + uuid4().hex
    command_base = ["runc", "--root", str(runtime / "state"), "--rootless", "true"]
    cleanup = []
    result: dict[str, Any] = {}
    try:
        staged = stage_candidate_package(candidate, runtime / "candidate-package")
        run_root = runtime / "rootfs"
        shutil.copytree(base, run_root, symlinks=True, copy_function=_copy_root_file)
        config = root_receipt["image_config_metadata"]
        environment = config.get("Env", [])
        path_env = dict(item.split("=", 1) for item in environment if "=" in item).get(
            "PATH", "/usr/local/bin:/usr/bin:/bin"
        )
        python = next(
            (
                posixpath.join(folder, name)
                for folder in path_env.split(":")
                if folder.startswith("/")
                for name in ("python", "python3")
                if _container_path(run_root, posixpath.join(folder, name)).is_file()
            ),
            None,
        )
        if python is None:
            raise ValueError("image has no Python interpreter on its declared PATH")
        spec = _build_spec(run_root, staged.root, inputs, output, python=python, env=environment, device=device)
        if plan:
            plan = _snapshot_cdi(plan, runtime / "cdi-dependencies")
            _apply_cdi(spec, plan)
            atomic_json(runtime / "cdi-plan.json", plan)
        for mount in spec["mounts"]:
            directory = mount["type"] != "bind" or Path(mount["source"]).is_dir()
            _mountpoint(run_root, mount["destination"], directory=directory)
        run_before, records = tree_manifest(run_root)
        atomic_json(runtime / "rootfs-before.json", {"sha256": run_before, "entries": records})
        atomic_json(runtime / "config.json", spec)
        args = command_base + [
            "--log",
            str(runtime / "runc.log"),
            "run",
            "--bundle",
            str(runtime),
            "--pid-file",
            str(runtime / "init.pid"),
            instance,
        ]

        def kill_all():
            _runc_action(command_base, ["kill", "--all", instance, "KILL"], "kill_all", cleanup)

        cdi_before = _verify_cdi_snapshot(plan) if plan else None
        try:
            result = _execute(
                args,
                runtime / "stdout.log",
                runtime / "stderr.log",
                timeout_seconds=timeout_seconds,
                on_timeout=kill_all,
            )
        finally:
            result.update(_cleanup_runtime(command_base, instance, runtime, cleanup))
        cdi_after = _verify_cdi_snapshot(plan) if plan else None
        run_after, _ = tree_manifest(run_root)
        base_after, _ = tree_manifest(base)
        candidate.verify()
        staged.verify()
        if input_hash != file_sha256(inputs):
            raise ValueError("input file changed during execution")
        if run_after != run_before or base_after != root_receipt["rootfs_tree_sha256"]:
            raise ValueError("rootfs tree changed during execution")
        result.update(
            schema="passage-rootless-oci-run/v1",
            candidate_identity=candidate.identity,
            manifest_sha256=candidate.digest,
            input_file_sha256=input_hash,
            rootfs_receipt_sha256=file_sha256(receipt_path),
            image_inspect_id=root_receipt["image_inspect_id"],
            runtime_directory=str(runtime),
            config_sha256=file_sha256(runtime / "config.json"),
            rootfs_tree_before_sha256=run_before,
            rootfs_tree_after_sha256=run_after,
            base_rootfs_tree_after_sha256=base_after,
            rootfs_unchanged=True,
            cleanup=cleanup,
            cdi_dependency_checks={"before": cdi_before, "after": cdi_after},
            device=device,
            gpu_uuid=gpu_uuid,
            cdi=plan,
            resource_cgroup_enforced=False,
            strict_admission=False,
        )
        atomic_json(output / "oci-run.json", result)
        if result["cleanup_errors"]:
            raise RuntimeError("runtime cleanup commands failed; inspect accumulated cleanup errors")
        if result["child_namespace_drained"] is not True:
            raise RuntimeError("private runtime drainage failed or could not be established")
        if result["returncode"] == 0 and not result["timed_out"]:
            from passage_eval.campaign.candidate.binding import bind_container_evidence

            bind_container_evidence(output)
        return result
    except BaseException as error:
        atomic_json(
            runtime / "failure.json",
            {"error": f"{type(error).__name__}: {error}", "result": result, "cleanup": cleanup},
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--image", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--rootfs", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--gpu-uuid")
    run.add_argument("--timeout-seconds", type=float, default=300)
    cdi = commands.add_parser("cdi-plan")
    cdi.add_argument("--gpu-uuid", required=True)
    cdi.add_argument("--spec", type=Path, default=Path("/var/run/cdi/nvidia.yaml"))
    cdi.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_rootfs(args.image, args.output)
    elif args.command == "cdi-plan":
        result = prepare_cdi(args.gpu_uuid, args.spec)
        atomic_json(args.output, result)
    else:
        result = run_oci(
            args.rootfs,
            args.manifest,
            args.inputs,
            args.output,
            device=args.device,
            gpu_uuid=args.gpu_uuid,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("returncode", 0) == 0 and not result.get("timed_out") else 1


if __name__ == "__main__":
    raise SystemExit(main())
