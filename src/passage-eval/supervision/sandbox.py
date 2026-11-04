# MIT License
#
# Copyright (c) 2026 THALES

"""Worker launch hardening: scrubbed environment, rlimits, optional ``bwrap``.

What each isolation level really provides is recorded by :func:`isolation_record`
and should be quoted from there, not paraphrased:

``process``
    a separate process in its own session and process group, with a scrubbed
    environment, a private temporary HOME/cwd and rlimits. It is the *same uid*
    on the *same filesystem and network*: the candidate can read any file this
    user can read and open sockets. This is a fault-containment and
    accidental-leak boundary, not a security boundary. **Candidate code runs
    unsandboxed**, and so does everything the candidate interpreter executes at
    start-up (``.pth`` files, ``sitecustomize``), including the interpreter probe.

``bwrap``
    additionally an unprivileged bubblewrap sandbox: user/ipc/pid/net/uts/cgroup
    namespaces unshared (loopback only, no route to the host network), a tmpfs
    root that contains only read-only binds of the interpreter environment and of
    private staged copies of the declared candidate artifacts, tmpfs HOME and /tmp.
    It is not a hardened boundary against kernel exploits, does not cap GPU memory,
    applies no seccomp filter and sets no cgroup CPU/memory limits.

The candidate interpreter is never executed outside the sandbox that the worker will use:
:func:`static_runtime_layout` derives the interpreter's bind set from files only (``pyvenv.cfg``,
``.pth`` files and setuptools editable finders are *read*, not run), and
:func:`probe_interpreter` then runs the probe inside that same sandbox. Anything the probe reports
that is not covered by a bind makes the run fail closed.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import resource
import shutil
import select
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Variables copied from the supervisor environment when present.
ENV_ALLOW_EXACT = frozenset(
    {
        "PATH",
        "LANG",
        "LANGUAGE",
        # thread-count controls
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        # accelerator device selection (selection only, no credentials)
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "OPENVINO_TORCH_BACKEND_DEVICE",
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
    }
)
ENV_ALLOW_PREFIX = ("LC_",)
#: Names that are never forwarded, not even as explicit extras.
_FORBIDDEN_EXTRA = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|PRIVATE|^AWS_|^AZURE_|^GOOGLE_|^GCP_|^SSH_|"
    r"^GITHUB|^GH_|^HF_|^HUGGING|^WANDB|^MLFLOW|^PYTHON|^LD_PRELOAD$|^LD_AUDIT$)",
    re.IGNORECASE,
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SANDBOX_HOME = "/home/worker"

#: Credential-bearing locations that no bind may expose (relative to the user's home).
DEFAULT_PROTECTED_HOME_ENTRIES = (
    ".ssh",
    ".aws",
    ".config",
    ".cache/huggingface",
    ".huggingface",
    ".netrc",
    ".gnupg",
    ".docker",
    ".kube",
    ".git-credentials",
    ".local/share/keyrings",
)
_SYSTEM_SYMLINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")


class SandboxError(RuntimeError):
    """The requested isolation cannot be established. Never downgraded silently."""


def check_device_node(device: str) -> str:
    """A device bind must be an existing character-device node below ``/dev`` and nothing else.

    ``--dev-bind`` gives the sandbox read-write access to whatever it names. ``/dev/shm`` (a directory) would become a
    shared writable directory, a symbolic link such as ``/dev/stdout`` would resolve to something else, and a regular
    file or a block device is not a device the benchmark needs. The node is checked without following links, and no
    component of the path may be a link.
    """
    if not isinstance(device, str) or not device.startswith("/dev/") or ".." in Path(device).parts:
        raise ValueError(f"device binds must be paths below /dev, got {device!r}")
    try:
        info = os.lstat(device)
    except OSError as error:
        raise ValueError(f"device bind {device!r} does not exist ({error.strerror or error})") from None
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"device bind {device!r} is a symbolic link; name the character-device node itself")
    if not stat.S_ISCHR(info.st_mode):
        kind = "a directory" if stat.S_ISDIR(info.st_mode) else "not a character device"
        raise ValueError(f"device bind {device!r} is {kind}; only existing character-device nodes may be bound")
    if os.path.realpath(device) != os.path.normpath(device):
        raise ValueError(f"device bind {device!r} passes through a symbolic link ({os.path.realpath(device)})")
    return device


@dataclass(frozen=True)
class RlimitConfig:
    """Limits applied by the worker to itself before any candidate code runs.

    ``core`` and ``nofile`` are always applied; the others are optional. The values
    actually in force are read back (``getrlimit``) and reported in the manifest.
    ``cpu_seconds`` is cumulative CPU time of one worker process, not per request.
    """

    core: int = 0
    nofile: int = 1024
    fsize_bytes: int | None = 1 << 30
    address_space_bytes: int | None = None
    cpu_seconds: int | None = None

    def __post_init__(self):
        if self.core != 0:
            raise ValueError("core dumps must stay disabled (core=0)")
        for name in ("nofile", "fsize_bytes", "address_space_bytes", "cpu_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")

    def worker_args(self) -> list[str]:
        args = ["--rlimit-core", "0", "--rlimit-nofile", str(self.nofile)]
        for flag, value in (
            ("--rlimit-fsize", self.fsize_bytes),
            ("--rlimit-as", self.address_space_bytes),
            ("--rlimit-cpu", self.cpu_seconds),
        ):
            if value is not None:
                args += [flag, str(value)]
        return args


@dataclass(frozen=True)
class IsolationConfig:
    level: str = "process"  # 'process' | 'bwrap'
    ro_binds: tuple[str, ...] = ()  # declared candidate artifact paths (files or directories; staged)
    runtime_binds: tuple[str, ...] = ()  # extra interpreter-environment trees (bound and hashed in place)
    dev_binds: tuple[str, ...] = ()  # device nodes, e.g. /dev/nvidia0 (opt-in, bwrap only)
    protected_paths: tuple[str, ...] = ()  # paths no bind may expose (dataset, references, registry)
    env_set: Mapping[str, str] = field(default_factory=dict)  # explicit extras
    env_passthrough: tuple[str, ...] = ()  # explicit extras copied from the supervisor environment
    bwrap_path: str = "bwrap"
    rlimits: RlimitConfig = field(default_factory=RlimitConfig)

    def __post_init__(self):
        if self.level not in {"process", "bwrap"}:
            raise ValueError("isolation level must be 'process' or 'bwrap'")
        if self.level == "process" and self.dev_binds:
            # Device binds need a mount namespace; refuse rather than imply a capability.
            raise ValueError("dev_binds require the bwrap isolation level")
        for device in self.dev_binds:
            check_device_node(device)


def _check_env_name(name: str) -> None:
    if not isinstance(name, str) or not _ENV_NAME.match(name):
        raise ValueError(f"invalid environment variable name {name!r}")
    if _FORBIDDEN_EXTRA.search(name):
        raise ValueError(f"refusing to forward credential-like or interpreter-control variable {name!r}")


def scrubbed_env(
    source: Mapping[str, str],
    *,
    home: str,
    tmpdir: str,
    extras: Mapping[str, str] | None = None,
    passthrough: Iterable[str] = (),
) -> dict[str, str]:
    """Allow-list environment for a worker; everything else is dropped."""
    env: dict[str, str] = {}
    for name, value in source.items():
        if name in ENV_ALLOW_EXACT or name.startswith(ENV_ALLOW_PREFIX):
            env[name] = value
    for name in passthrough:
        _check_env_name(name)
        if name in source:
            env[name] = source[name]
    for name, value in (extras or {}).items():
        _check_env_name(name)
        if not isinstance(value, str):
            raise ValueError("explicit environment values must be strings")
        env[name] = value
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(HOME=home, TMPDIR=tmpdir, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


def _overlaps(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def default_protected_paths(extra: Iterable[str | Path] = ()) -> tuple[str, ...]:
    home = Path.home()
    paths = [home / entry for entry in DEFAULT_PROTECTED_HOME_ENTRIES] + [Path(p) for p in extra]
    return tuple(str(p) for p in paths)


def check_binds(binds: Iterable[str | Path], protected: Iterable[str | Path]) -> list[Path]:
    """Every bind must exist and must neither contain nor sit inside a protected path."""
    resolved = []
    for bind in binds:
        path = Path(bind)
        if not path.exists():
            raise SandboxError(f"declared bind does not exist: {path}")
        for guarded in protected:
            if _overlaps(path, Path(guarded)):  # also for paths that do not exist yet (e.g. the output dir)
                raise SandboxError(f"bind {path} overlaps protected path {guarded}")
        resolved.append(path.resolve())
    return resolved


@dataclass(frozen=True)
class Bind:
    """One read-only bind: the bytes at ``source`` appear at ``target`` inside the sandbox."""

    source: str
    target: str
    role: str = "runtime"  # runtime | candidate_staged


def as_bind(item: "Bind | str | Path") -> Bind:
    return item if isinstance(item, Bind) else Bind(str(item), str(item))


@dataclass(frozen=True)
class RuntimeLayout:
    """Facts about a worker interpreter that were read from files, without executing it."""

    python: str
    real_executable: str
    venv_root: str | None
    base_prefix: str
    site_dirs: tuple[str, ...]
    roots: tuple[str, ...]  # minimal set of trusted trees the sandbox binds for the interpreter
    pth_import_lines: int  # executable ``import`` lines in .pth files (run inside the sandbox)
    editable_roots: Mapping[str, str]  # import name -> directory, from setuptools editable finders
    pth_roots: tuple[str, ...]  # every directory the .pth files name (trusted or not)
    needs_declaration: tuple[
        tuple[str, str], ...
    ] = ()  # (path, reason): named by the environment, not bound on its say-so


_EDITABLE_IMPORT = re.compile(r"^import\s+(__editable__\w*finder)\s*;")


def _parse_cfg(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip().lower()] = value.strip()
    return values


def _editable_mapping(finder: Path) -> tuple[dict[str, str], list[str]]:
    """Read ``MAPPING`` / ``NAMESPACES`` from a setuptools editable finder with ``ast`` (nothing is executed)."""
    mapping: dict[str, str] = {}
    namespaces: list[str] = []
    try:
        tree = ast.parse(finder.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return mapping, namespaces
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target not in {"MAPPING", "NAMESPACES"} or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        if target == "MAPPING" and isinstance(literal, dict):
            mapping = {str(k): str(v) for k, v in literal.items() if isinstance(v, str)}
        if target == "NAMESPACES" and isinstance(literal, dict):
            for paths in literal.values():
                namespaces += [str(x) for x in paths if isinstance(x, str)]
    return mapping, namespaces


def _minimal_cover(paths: Iterable[Path]) -> list[Path]:
    kept: list[Path] = []
    for path in sorted({Path(p) for p in paths}, key=lambda item: (len(item.parts), str(item))):
        if not any(path == known or known in path.parents for known in kept):
            kept.append(path)
    return sorted(kept)


def evaluator_roots() -> list[Path]:
    """The source directories of the supervisor's own ``passage`` and ``passage_eval`` (trusted, not candidate-chosen)."""
    import passage
    import passage_eval

    return [Path(module.__file__).resolve().parent for module in (passage, passage_eval)]


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def static_runtime_layout(
    python: str | Path, extra_roots: Iterable[str | Path] = (), trusted_roots: Iterable[str | Path] = ()
) -> RuntimeLayout:
    """Locate what the worker interpreter needs, reading files only.

    The candidate controls the files of its virtual environment, so a path that such a file names is **not**
    bound on its say-so: a ``.pth`` line or an editable finder could otherwise expose any directory of the host.
    Bound automatically are ``/usr`` (loader, libc, standard library), the environment's own root,
    ``extra_roots`` (declared by the operator, e.g. ``--runtime-bind``), and the directories named by ``.pth``
    path lines and setuptools editable finders that lie inside the environment, inside ``/usr`` or inside
    ``trusted_roots`` (the supervisor's own evaluator sources, see :func:`evaluator_roots`). Everything else
    that the environment names, and an interpreter prefix outside ``/usr``, is returned in
    ``needs_declaration`` for the operator to declare. ``.pth`` lines that start with ``import`` are counted,
    not run: they execute inside the sandbox when the worker starts.
    """
    executable = Path(os.path.abspath(python))
    if not executable.exists():
        raise SandboxError(f"worker interpreter does not exist: {executable}")
    real = executable.resolve()
    config_file = next(
        (d / "pyvenv.cfg" for d in (executable.parent, executable.parent.parent) if (d / "pyvenv.cfg").is_file()), None
    )
    usr = Path("/usr")
    venv_root = config_file.parent.resolve() if config_file is not None else None
    # The prefix comes from the executable that will really run, never from the ``home`` line of pyvenv.cfg.
    base_prefix = real.parent.parent
    if venv_root is not None:
        site_dirs = sorted({site.resolve() for site in venv_root.glob("lib*/python*/site-packages")})
    else:
        site_dirs = sorted(
            {
                site.resolve()
                for pattern in ("lib*/python*/site-packages", "lib*/python*/dist-packages")
                for site in base_prefix.glob(pattern)
            }
        )
    declared = [Path(p).resolve() for p in extra_roots]
    for root in declared:
        if not root.exists():
            raise SandboxError(f"runtime root does not exist: {root}")
    trusted = [usr, *declared, *[Path(p).resolve() for p in trusted_roots]] + ([venv_root] if venv_root else [])
    roots: list[Path] = [usr, *declared] + ([venv_root] if venv_root else [])
    needs: list[tuple[str, str]] = []
    if base_prefix != Path("/") and not any(_under(base_prefix, root) for root in trusted):
        needs.append((str(base_prefix), "prefix of the interpreter executable, outside /usr"))
    imports, pth_roots = 0, []
    editable: dict[str, str] = {}
    for site in site_dirs:
        for pth in sorted(site.glob("*.pth")):
            try:
                lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    imports += 1
                    matched = _EDITABLE_IMPORT.match(line)
                    if matched:
                        mapping, namespaces = _editable_mapping(site / f"{matched.group(1)}.py")
                        editable.update(mapping)
                        pth_roots += [Path(v) for v in [*mapping.values(), *namespaces]]
                    continue
                candidate = Path(os.path.normpath(site / line))
                if candidate.is_dir():
                    pth_roots.append(candidate)
    resolved_pth = sorted({Path(p).resolve() for p in pth_roots if Path(p).exists()})
    for root in resolved_pth:
        if any(_under(root, known) for known in trusted):
            roots.append(root)
        else:
            needs.append((str(root), "named by a .pth file or editable finder of the environment"))
    return RuntimeLayout(
        python=str(executable),
        real_executable=str(real),
        venv_root=None if venv_root is None else str(venv_root),
        base_prefix=str(base_prefix),
        site_dirs=tuple(str(s) for s in site_dirs),
        roots=tuple(str(r) for r in _minimal_cover(roots)),
        pth_import_lines=imports,
        editable_roots=dict(sorted(editable.items())),
        pth_roots=tuple(str(r) for r in resolved_pth),
        needs_declaration=tuple(sorted(set(needs))),
    )


_PROBE_MAGIC = b"PASSAGE-PROBE/v2\n"
_PROBE_CODE = (
    "import json, os, sys, importlib.util\n"
    "fd = int(sys.argv[1]); extra = json.loads(sys.argv[2]); names = json.loads(sys.argv[3])\n"
    "base_path = [p for p in sys.path if p]\n"
    "sys.path.extend(extra)\n"
    "out = dict(executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,\n"
    "           exec_prefix=sys.exec_prefix, base_exec_prefix=sys.base_exec_prefix,\n"
    "           version=list(sys.version_info[:3]), sys_path=base_path, modules={})\n"
    "for name in names + ['passage_eval', 'passage', 'numpy']:\n"
    "    spec = importlib.util.find_spec(name)\n"
    "    out['modules'][name] = None if spec is None else dict(\n"
    "        origin=spec.origin, locations=list(spec.submodule_search_locations or []))\n"
    "payload = json.dumps(out).encode()\n"
    "data = " + repr(_PROBE_MAGIC) + " + str(len(payload)).encode() + b'\\n' + payload\n"
    "view = memoryview(data)\n"
    "while view:\n"
    "    view = view[os.write(fd, view):]\n"
    "os.close(fd)\n"
)
_PROBE_OUTPUT_LIMIT = 1 << 20
_PROBE_PIPE_LIMIT = _PROBE_OUTPUT_LIMIT + 64
_STARTUP_OUTPUT_CHARS = 1024
#: Bytecode written or read by a sandboxed interpreter lives here (a tmpfs inside the sandbox).
SANDBOX_PYCACHE = "/tmp/passage-pycache"


def pycache_args(prefix: str) -> list[str]:
    """``-B -X pycache_prefix=...``: bytecode is looked up in a fresh private directory only, and never written.

    Without it an interpreter loads ``__pycache__/*.pyc`` that lie beside the sources it imports, and a ``.pyc`` whose
    recorded source mtime and size match is trusted. Trees hashed in place (``process`` isolation, the interpreter
    environment) are hashed as sources and files, but nothing relates a stale or forged ``.pyc`` in them to the source
    that was hashed. A prefix that is empty at start makes every import compile from source. ``-B`` is explicit because
    ``-I`` also ignores ``PYTHONDONTWRITEBYTECODE``: without it a worker would write bytecode into the prefix (or, before
    the prefix existed, beside the sources of the trees that were hashed).
    """
    return ["-B", "-X", f"pycache_prefix={prefix}"]


def _limit_probe_output():  # pragma: no cover - runs in the child before exec
    resource.setrlimit(resource.RLIMIT_FSIZE, (_PROBE_OUTPUT_LIMIT, _PROBE_OUTPUT_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def parse_probe_message(data: bytes) -> tuple[dict, int]:
    """The FIRST complete probe message of ``data`` and the number of bytes that follow it (ignored, never parsed)."""
    if not data.startswith(_PROBE_MAGIC):
        raise SandboxError("the probe channel does not start with a probe message")
    rest = data[len(_PROBE_MAGIC) :]
    newline = rest.find(b"\n")
    header = rest[:newline] if newline >= 0 else b""
    if not 1 <= len(header) <= 9 or not header.isdigit():
        raise SandboxError("the probe message has no valid length")
    length = int(header)
    if length > _PROBE_OUTPUT_LIMIT:
        raise SandboxError("the probe message is larger than the limit")
    body = rest[newline + 1 : newline + 1 + length]
    if len(body) < length:
        raise SandboxError("the probe message is incomplete")
    try:
        probe = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SandboxError("the probe message is not valid JSON") from None
    if not isinstance(probe, dict) or not isinstance(probe.get("modules"), dict):
        raise SandboxError("the probe message has no module table")
    return probe, len(rest) - (newline + 1 + length)


def _drain_probe(proc: subprocess.Popen, descriptor: int, timeout: float) -> bytes:
    """Read the probe pipe until the interpreter exits (its descendants may keep the write end open: EOF is not awaited)."""
    chunks, total = [], 0
    deadline = time.monotonic() + timeout
    os.set_blocking(descriptor, False)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            with_group_kill(proc)
            raise SandboxError(f"interpreter probe timed out after {timeout:.0f} s")
        ready, _, _ = select.select([descriptor], [], [], min(remaining, 0.05))
        if ready:
            try:
                block = os.read(descriptor, 1 << 16)
            except BlockingIOError:
                block = b""
            if block:
                chunks.append(block)
                total += len(block)
                if total > _PROBE_PIPE_LIMIT:
                    with_group_kill(proc)
                    raise SandboxError("the interpreter probe wrote more than the limit to its result channel")
                continue
        if proc.poll() is not None:
            try:  # whatever was written before the exit is still in the pipe
                while True:
                    block = os.read(descriptor, 1 << 16)
                    if not block:
                        break
                    chunks.append(block)
                    total += len(block)
                    if total > _PROBE_PIPE_LIMIT:
                        raise SandboxError("the interpreter probe wrote more than the limit to its result channel")
            except BlockingIOError:
                pass
            return b"".join(chunks)


def probe_interpreter(
    python: str | Path,
    *,
    isolation: "IsolationConfig | None" = None,
    binds: Sequence["Bind | str | Path"] = (),
    candidate_paths: Sequence[str | Path] = (),
    top_levels: Sequence[str] = (),
    timeout: float = 60.0,
) -> dict:
    """Ask the worker interpreter where its runtime lives, *inside the sandbox the worker will use*.

    The probe runs ``python -I -c ...`` under exactly the isolation of the worker: in ``bwrap`` mode the same
    read-only binds and namespaces (so a ``.pth`` file or ``sitecustomize`` in the candidate environment can
    see nothing the worker could not see), in ``process`` mode with the same scrubbed environment and no
    confinement (candidate code is unsandboxed at this level, and so is its start-up code).
    ``-S`` is deliberately not used: the probe must observe the ``sys.path`` the worker will have, and that
    includes entries added by ``.pth`` files and editable finders. ``find_spec`` on a top-level name runs path
    finders only; it does not import the module.

    The result travels on a **dedicated pipe descriptor** that the supervisor creates and passes to the interpreter
    (it survives ``bwrap``), not on standard output. Only the FIRST complete message is used: an exit hook (``atexit``
    in a ``.pth`` file or ``sitecustomize``) that writes a second message, or text that looks like a marker on standard
    output, changes nothing. Whatever follows the first message is counted and recorded
    (``channel.trailing_bytes_ignored``). A hook that runs before the probe and writes first can still pre-empt it;
    the defence against that is to pin the interpreter environment (``--environment-allow-sha256``), which a
    verified run requires.
    """
    isolation = isolation or IsolationConfig()
    private = Path(tempfile.mkdtemp(prefix="passage-probe-"))
    read_end = write_end = None
    try:
        os.chmod(private, 0o700)
        in_sandbox = isolation.level == "bwrap"
        home = SANDBOX_HOME if in_sandbox else str(private / "home")
        tmpdir = "/tmp" if in_sandbox else str(private / "tmp")
        if not in_sandbox:
            os.mkdir(home, 0o700)
            os.mkdir(tmpdir, 0o700)
        prefix = SANDBOX_PYCACHE if in_sandbox else str(private / "pycache")
        env = scrubbed_env(
            os.environ, home=home, tmpdir=tmpdir, extras=isolation.env_set, passthrough=isolation.env_passthrough
        )
        read_end, write_end = os.pipe()
        os.set_inheritable(write_end, True)
        inner = [
            os.path.abspath(python),
            "-I",
            *pycache_args(prefix),
            "-c",
            _PROBE_CODE,
            str(write_end),
            json.dumps([str(Path(p).resolve()) for p in candidate_paths]),
            json.dumps(list(top_levels)),
        ]
        argv = bwrap_command(isolation, [as_bind(b) for b in binds], inner) if in_sandbox else inner
        out_path, err_path = private / "stdout", private / "stderr"
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                env=env,
                cwd=str(private) if in_sandbox else home,
                start_new_session=True,
                close_fds=True,
                pass_fds=(write_end,),
                preexec_fn=_limit_probe_output,
            )
            os.close(write_end)
            write_end = None
            try:
                channel = _drain_probe(proc, read_end, timeout)
            finally:
                if proc.poll() is None:
                    with_group_kill(proc)
        stdout = out_path.read_bytes().decode("utf-8", "replace")
        if proc.returncode != 0:
            tail = err_path.read_bytes().decode("utf-8", "replace").strip()[-400:]
            raise SandboxError(f"cannot probe worker interpreter (exit {proc.returncode}): {tail}")
        if not channel:
            raise SandboxError("worker interpreter probe produced no result")
        probe, trailing = parse_probe_message(channel)
        startup = stdout.strip()[:_STARTUP_OUTPUT_CHARS]
    finally:
        for descriptor in (read_end, write_end):
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        shutil.rmtree(private, ignore_errors=True)
    for required in ("passage_eval", "passage", "numpy"):
        if probe["modules"].get(required) is None:
            raise SandboxError(
                f"worker interpreter cannot import {required}"
                + (f" (the interpreter printed at start-up: {startup[:200]!r})" if startup else "")
            )
    probe["startup_output"] = startup  # what the interpreter wrote to stdout (should be empty); never parsed
    probe["executed_in"] = (
        "bwrap_same_sandbox_as_worker" if isolation.level == "bwrap" else "unsandboxed_process_level_same_as_worker"
    )
    probe["channel"] = {
        "kind": "dedicated_pipe_descriptor",
        "message_used": "first_complete_message_only",
        "trailing_bytes_ignored": trailing,
        "stdout_never_parsed": True,
    }
    return probe


def with_group_kill(proc: subprocess.Popen) -> None:
    import contextlib

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=15)


def probe_paths(probe: Mapping[str, Any]) -> list[Path]:
    """Every existing file system location the probe reports (sys.path, prefixes, module locations)."""
    found: list[str] = [
        probe["prefix"],
        probe["base_prefix"],
        probe["exec_prefix"],
        probe["base_exec_prefix"],
        *probe["sys_path"],
    ]
    for module in probe["modules"].values():
        if module is not None:
            found += [module["origin"] or "", *module["locations"]]
    return [Path(p) for p in dict.fromkeys(found) if p and os.path.isabs(p) and os.path.exists(p)]


def uncovered_probe_paths(probe: Mapping[str, Any], covered: Iterable[str | Path]) -> list[Path]:
    """Reported locations that are neither inside a bound tree nor equal to one (sorted)."""
    roots = [Path(c).resolve() for c in covered]
    missing = []
    for path in probe_paths(probe):
        real = path.resolve()
        if not any(real == root or root in real.parents for root in roots):
            missing.append(path)
    return sorted(missing)


def isolation_record(
    config: IsolationConfig, binds: Sequence["Bind | Path | str"], *, bwrap_version: str | None
) -> dict:
    """The manifest description of the isolation actually requested/applied (no more)."""
    base = {
        "level": config.level,
        "effective_level": config.level,
        "same_uid_as_supervisor": True,
        "own_session_and_process_group": True,
        "environment": "allow_list_scrubbed",
        "private_home_and_cwd": True,
        "not_applied": [
            "gpu_memory_cap",
            "seccomp_filter",
            "cgroup_cpu_memory_limits",
            "protection_against_kernel_exploits",
        ],
    }
    if config.level == "process":
        base.update(
            filesystem="shared_with_supervisor_user",
            network="shared_with_host",
            namespaces=[],
            read_only_binds=[],
            device_binds=[],
            candidate_code_confined=False,
            candidate_code_runs_unsandboxed=True,
            interpreter_probe="executed_unsandboxed_process_level_same_as_worker",
            statement=(
                "process isolation: candidate code, and the code the candidate interpreter runs at "
                "start-up (.pth files, sitecustomize, the interpreter probe), run unsandboxed as the "
                "supervisor's user with the supervisor's file system and network; this is fault "
                "containment, not a security boundary, and it is not admissible for a verified run"
            ),
        )
        base["not_applied"] += ["filesystem_isolation", "network_isolation", "pid_isolation"]
    else:
        converted = [as_bind(b) for b in binds]
        base.update(
            filesystem="tmpfs_root_with_read_only_binds",
            network="unshared_loopback_only",
            namespaces=["user", "ipc", "pid", "net", "uts", "cgroup-try"],
            read_only_binds=[b.target for b in converted],
            read_only_bind_sources=[
                {
                    "target": b.target,
                    "role": b.role,
                    "source": "private_staged_copy" if b.role == "candidate_staged" else "host_in_place",
                }
                for b in converted
            ],
            device_binds=list(config.dev_binds),
            device_bind_policy="existing_character_device_nodes_only_no_links_no_directories",
            tmpfs=["/tmp", SANDBOX_HOME],
            bwrap_version=bwrap_version,
            die_with_parent=True,
            candidate_code_confined=True,
            candidate_code_runs_unsandboxed=False,
            interpreter_probe="executed_in_bwrap_same_sandbox_as_worker",
            statement=(
                "bwrap isolation: candidate code and the interpreter probe run in an unprivileged "
                "bubblewrap sandbox; not a hardened boundary (kernel exploits, no seccomp, no cgroup "
                "or GPU memory limits)"
            ),
        )
    return base


def bwrap_version(bwrap_path: str) -> str:
    resolved = shutil.which(bwrap_path)
    if resolved is None:
        raise SandboxError(f"bwrap not found: {bwrap_path!r}; refusing to fall back to weaker isolation")
    try:
        completed = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise SandboxError(f"bwrap unusable: {error}") from None
    return completed.stdout.strip() or "unknown"


def bwrap_command(config: IsolationConfig, binds: Sequence["Bind | Path | str"], inner: Sequence[str]) -> list[str]:
    """Build the bubblewrap argv. The root is a tmpfs; nothing is bound unless listed here."""
    argv = [
        shutil.which(config.bwrap_path) or config.bwrap_path,
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for system_path in _SYSTEM_SYMLINKS:
        link = Path(system_path)
        if link.is_symlink():
            argv += ["--symlink", os.readlink(link), system_path]
    if Path("/etc/ld.so.cache").exists():
        argv += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", SANDBOX_HOME]
    # runtime trees first, staged candidate copies last: a staged copy shadows what lies under a runtime tree
    ordered = sorted((as_bind(b) for b in binds), key=lambda b: (b.role == "candidate_staged", b.target))
    for bind in ordered:
        if bind.target == "/usr" and bind.source == "/usr":
            continue
        argv += ["--ro-bind", bind.source, bind.target]
    for device in config.dev_binds:
        argv += ["--dev-bind", check_device_node(device), device]  # checked again: the node may have been replaced
    argv += ["--chdir", SANDBOX_HOME, "--", *inner]
    return argv


def available_bwrap(path: str = "bwrap") -> bool:
    """True when an unprivileged sandbox can really be created here (used by tests)."""
    resolved = shutil.which(path)
    if resolved is None:
        return False
    try:
        completed = subprocess.run(
            [
                resolved,
                "--unshare-all",
                "--die-with-parent",
                "--ro-bind",
                "/usr",
                "/usr",
                *sum([["--symlink", os.readlink(p), p] for p in _SYSTEM_SYMLINKS if Path(p).is_symlink()], []),
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--",
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
