# MIT License
#
# Copyright (c) 2026 THALES

"""Candidate identity measured from file bytes, never by importing candidate code.

``passage_eval.cli._candidate_identity`` imports the entry point and hashes loaded
code objects; the supervisor must not do that. Here the files that *will* be
executed are located with the worker interpreter's own path finders
(``importlib.util.find_spec`` on the top-level name only, run inside the worker's
sandbox) and hashed. The worker later reports the files it actually loaded; a file
that is not in the hashed set is a fatal identity mismatch (:func:`check_loaded_files`),
but that report is self-reported and only a cross-check.

What makes the identity cover the run is the set of **bound trees** recorded beside the
per-module hashes: every declared candidate file or directory is staged into a private
read-only copy that is the only thing the sandbox binds (hashed while it is copied, so
nothing is hashed that is not used), and the interpreter environment (virtual environment,
base prefix, editable source roots, ``/usr``) is hashed in place. Each tree carries a
deterministic digest (:mod:`passage_eval.supervision.treehash`). Strings in the inference
kwargs that name existing files outside those trees are refused
(:func:`check_kwargs_paths`).

The result uses ``schema = passage-candidate-identity-v1`` so that the existing
offline scorer accepts it, with ``identity_method = tree_hash_of_bound_bytes_without_import``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from passage_eval.execution.artifacts import sha256_file
from passage_eval.supervision import treehash

SCHEMA = "passage-candidate-identity-v1"
IDENTITY_METHOD = "tree_hash_of_bound_bytes_without_import"
BINDINGS = ("staged_read_only_copy", "in_place_hash_rechecked")
_CHECKPOINT_KEY_HINTS = ("checkpoint", "bundle")
#: Strings that are shaped like a file system path (absolute, home-relative or explicitly relative).
_PATHLIKE = ("/", "~/", "./", "../")


class IdentityError(ValueError):
    """The declared candidate files cannot be located or verified."""


@dataclass
class PathView:
    """Maps an original path to the bytes that are actually bound (the staged copy, when there is one)."""

    mapping: list[tuple[Path, Path]] = field(default_factory=list)  # (original root, staged root)

    def add(self, original: str | Path, staged: str | Path) -> None:
        self.mapping.append((Path(original), Path(staged)))

    def __call__(self, path: str | Path) -> Path:
        path = Path(path)
        for original, staged in self.mapping:
            if path == original:
                return staged
            if original in path.parents:
                return staged / path.relative_to(original)
        return path

    def covers(self, path: str | Path) -> bool:
        path = Path(path)
        return any(path == original or original in path.parents for original, _ in self.mapping)


def split_entry(entry: str) -> tuple[str, str]:
    if not isinstance(entry, str) or entry.count(":") != 1:
        raise IdentityError(f"entry point must be 'module:attr', got {entry!r}")
    module, attr = entry.split(":")
    if not module or not attr or any(not part.isidentifier() for part in module.split(".")):
        raise IdentityError(f"invalid entry point {entry!r}")
    return module, attr


def top_level_names(entries: Sequence[str]) -> list[str]:
    return sorted({split_entry(entry)[0].split(".")[0] for entry in entries})


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def module_files(module: str, probe: Mapping[str, Any]) -> list[tuple[str, Path]]:
    """Files executed by ``import module``: ``(module name, file)`` for the package chain and the module."""
    parts = module.split(".")
    top = probe["modules"].get(parts[0])
    if top is None:
        raise IdentityError(f"cannot locate top-level module {parts[0]!r} without importing it")
    origin = Path(top["origin"]) if top["origin"] else None
    if len(parts) == 1:
        if origin is None or not origin.is_file():
            raise IdentityError(f"module {module!r} has no readable file (namespace or built-in)")
        return [(module, origin.resolve())]
    found: list[tuple[str, Path]] = []
    if origin is not None and origin.is_file():
        found.append((parts[0], origin.resolve()))  # top-level package __init__
    for location in top["locations"]:
        root = Path(location)
        chain = []
        for index in range(1, len(parts) - 1):
            init = root.joinpath(*parts[1 : index + 1]) / "__init__.py"
            if init.is_file():
                chain.append((".".join(parts[: index + 1]), init.resolve()))
        base = root.joinpath(*parts[1:-1])
        leaf = _existing(base / f"{parts[-1]}.py") or _existing(base / parts[-1] / "__init__.py")
        if leaf is None:
            for suffix in (".so", ".pyd"):
                matches = sorted(base.glob(f"{parts[-1]}.*{suffix}"))
                if matches:
                    leaf = matches[0]
                    break
        if leaf is not None:
            return found + chain + [(module, leaf.resolve())]
    raise IdentityError(f"cannot locate files of module {module!r} without importing it")


def checkpoint_files(kwargs: Mapping[str, Any]) -> dict[str, Path]:
    """Declared checkpoint/bundle artifacts (same rule as the in-process CLI)."""
    found = {}
    for key, value in kwargs.items():
        if any(hint in key for hint in _CHECKPOINT_KEY_HINTS):
            if not isinstance(value, str) or not Path(value).is_file():
                raise IdentityError(f"declared {key} must identify a readable local artifact: {value!r}")
            found[key] = Path(value)
    return found


def _strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def kwargs_paths(kwargs: Mapping[str, Any]) -> list[tuple[str, Path]]:
    """Every string in ``kwargs`` (nested too) that is shaped like a path and exists on this host."""
    found = []
    for key, text in _strings(kwargs):
        if text.startswith(_PATHLIKE) and len(text) > 1:
            candidate = Path(os.path.expanduser(text))
            if not candidate.is_absolute():
                candidate = Path(os.path.abspath(candidate))
            if os.path.lexists(candidate):
                found.append((key, candidate))
    return found


def check_kwargs_paths(role: str, kwargs: Mapping[str, Any], covered: Sequence[str | Path]) -> list[str]:
    """Refuse kwargs that point to existing files outside the declared (staged) and runtime trees.

    A weights file passed under a key such as ``onnx_path`` would otherwise be read by the candidate
    without ever being hashed. Declare it (``--ro-bind``) or give it a checkpoint/bundle key. Returns the
    key paths that were found inside the trees (for the manifest).
    """
    roots = [Path(c) for c in covered]
    inside, outside = [], []
    for key, candidate in kwargs_paths(kwargs):
        resolved = Path(os.path.realpath(candidate))
        if any(resolved == root or root in resolved.parents for root in roots):
            inside.append(key)
        else:
            outside.append(f"{key}={str(candidate)!r}")
    if outside:
        raise IdentityError(
            f"{role} kwargs point to existing files outside the declared candidate and runtime trees "
            f"({'; '.join(outside)}); declare them with --ro-bind or --candidate-path so that they are staged and hashed"
        )
    return inside


def declared_bundle_members(manifest_path: str | None) -> list[Path]:
    """Paths named by a ``passage-runtime-bundle-v1`` manifest (read in place; nothing is hashed here)."""
    if manifest_path is None:
        return []
    path = Path(manifest_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IdentityError(f"runtime bundle manifest unreadable: {error}") from None
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("files"), list)
        or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in payload["files"])
    ):
        raise IdentityError("runtime bundle manifest requires passage-runtime-bundle-v1 and nonempty files")
    return [path] + [(path.parent / item["path"]).resolve() for item in payload["files"]]


def runtime_bundle(manifest_path: str | None, view: Callable[[Path], Path] = Path) -> dict[str, Any] | None:
    """Verify a ``passage-runtime-bundle-v1`` file list against its recorded hashes.

    Bytes are read through ``view`` (the staged copy), so the verified bytes are the bound bytes.
    """
    if manifest_path is None:
        return None
    path = Path(manifest_path).resolve()
    payload = json.loads(view(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "passage-runtime-bundle-v1"
        or not isinstance(payload.get("files"), list)
        or not payload["files"]
    ):
        raise IdentityError("runtime bundle manifest requires passage-runtime-bundle-v1 and nonempty files")
    files = {}
    for item in payload["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["path"], str):
            raise IdentityError("runtime bundle files require path and sha256")
        member = (path.parent / item["path"]).resolve()
        if not view(member).is_file() or member.as_posix() in files or sha256_file(view(member)) != item["sha256"]:
            raise IdentityError("runtime bundle file identity mismatch or duplicate file")
        files[member.as_posix()] = item["sha256"]
    return {"path": str(path), "manifest_sha256": sha256_file(view(path)), "files": files}


def measure_component(
    entry: str, kwargs: Mapping[str, Any], probe: Mapping[str, Any], view: Callable[[Path], Path] = Path
) -> dict[str, Any]:
    """Hashes of one worker's entry module files and its declared checkpoint files (read through ``view``)."""
    module, _ = split_entry(entry)
    files, modules = {}, {}
    for name, path in module_files(module, probe):
        digest = sha256_file(view(path))
        files[path.as_posix()] = digest
        modules[name] = {"path": path.as_posix(), "sha256": digest}
    checkpoints = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(view(path.resolve()))}
        for key, path in checkpoint_files(kwargs).items()
    }
    return {"entry": entry, "module": module, "files": files, "modules": modules, "checkpoints": checkpoints}


def entry_package_path(entry: str, probe: Mapping[str, Any]) -> Path:
    """The top-level package directory (or single module file) that provides ``entry``.

    A namespace package (a directory without ``__init__.py``, or one that the interpreter resolves to several
    locations) is refused: its content is the union of directories that no single tree digest describes, and a
    portion can be added from any ``sys.path`` entry without touching the reviewed one.
    """
    module, _ = split_entry(entry)
    top = module.split(".")[0]
    found = probe["modules"].get(top)
    if found is None:
        raise IdentityError(f"cannot locate top-level module {top!r} without importing it")
    origin = Path(found["origin"]) if found["origin"] else None
    locations = found["locations"]
    if locations:
        if origin is None or origin.name != "__init__.py" or not origin.is_file():
            raise IdentityError(
                f"{top!r} is a namespace package (no __init__.py); a pinned entry point must be a regular "
                f"package or a single module, not a union of directories"
            )
        if len({str(Path(location).resolve()) for location in locations}) != 1:
            raise IdentityError(
                f"package {top!r} resolves to several locations {sorted(locations)}; refusing to pin it"
            )
        return origin.parent.resolve()
    if origin is None or not origin.is_file():
        raise IdentityError(f"module {top!r} has no readable file (namespace or built-in)")
    return origin.resolve()


def entry_tree(
    entry: str, probe: Mapping[str, Any], view: Callable[[Path], Path] = Path, role: str = "decoder_entry"
) -> treehash.TreeRecord:
    """Tree digest of the package (or module file) an entry point lives in, read through ``view``.

    This is the value an operator pins with ``--decoder-allow-sha256`` or ``--planner-allow-sha256``. It covers every
    helper module in that package and nothing else: a pinned worker is given exactly this tree, the interpreter
    environment and the evaluator sources (see :func:`passage_eval.supervision.environment.BoundSet.pinned_view`),
    so the digest is the whole candidate-side code it can import.
    """
    original = entry_package_path(entry, probe)
    record = treehash.hash_tree(
        view(original), role=role, symlinks="refuse", strict=True, exclude_dirs=treehash.CANDIDATE_EXCLUDES
    )
    record.path = str(original)
    return record


def decoder_entry_tree(
    entry: str, probe: Mapping[str, Any], view: Callable[[Path], Path] = Path
) -> treehash.TreeRecord:
    """Backward-compatible name of :func:`entry_tree` for a decoder."""
    return entry_tree(entry, probe, view, role="decoder_entry")


def build_identity(
    components: Mapping[str, Mapping[str, Any]],
    *,
    entry_label: str,
    bundle: dict | None,
    declared_source_sha256: str | None,
    bound_trees: Mapping[str, Any] | None = None,
    binding: str = "in_place_hash_rechecked",
) -> dict[str, Any]:
    """Merge component measurements into one ``passage-candidate-identity-v1`` record.

    ``bound_trees`` is the manifest form of every tree bound into the sandbox (see
    :mod:`passage_eval.supervision.treehash`); ``binding`` says whether the candidate bytes are a staged
    read-only copy or were hashed in place.
    """
    if binding not in BINDINGS:
        raise IdentityError(f"unknown identity binding {binding!r}")
    if declared_source_sha256 is not None and (
        not isinstance(declared_source_sha256, str)
        or len(declared_source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in declared_source_sha256)
    ):
        raise IdentityError("candidate_source_sha256 must be a lowercase 64-character SHA256 identity")
    modules: dict[str, dict[str, str]] = {}
    for component in components.values():
        for name, record in component["modules"].items():
            if modules.setdefault(name, record) != record:
                raise IdentityError(f"module {name!r} resolves to different files in different components")
    return {
        "schema": SCHEMA,
        "entry": entry_label,
        "defining_module": ",".join(sorted({c["module"] for c in components.values()})),
        "module_files": dict(sorted(modules.items())),
        "loaded_code_sha256": {},
        "runtime_bundle": bundle,
        "declared_source_sha256": declared_source_sha256,
        "source_verification": "file_content_measured",
        "identity_method": IDENTITY_METHOD,
        "identity_binding": binding,
        "bound_trees": dict(bound_trees or {}),
        "components": {role: dict(component) for role, component in sorted(components.items())},
    }


def hashed_file_set(identity: Mapping[str, Any]) -> set[str]:
    allowed = set()
    for component in identity["components"].values():
        allowed |= {os.path.realpath(path) for path in component["files"]}
    return allowed


def _tree_roots(identity: Mapping[str, Any]) -> list[str]:
    trees = identity.get("bound_trees") or {}
    return [os.path.realpath(row["path"]) for row in [*trees.get("candidate", []), *trees.get("environment", [])]]


def check_loaded_files(role: str, loaded: Sequence[str], identity: Mapping[str, Any]) -> None:
    """Fail closed if the worker executed a file that no identity covers.

    A file is covered when it is one of the component's individually hashed module files, or when it lies
    inside a bound tree whose digest is part of the identity (a helper, or the module that defines a class
    the entry module re-exports). The list comes from the candidate's own process: it is a cross-check
    of the static identity, not the authority for it.
    """
    allowed = {os.path.realpath(path) for path in identity["components"][role]["files"]}
    roots = _tree_roots(identity)
    outside = sorted(
        path
        for path in {os.path.realpath(p) for p in loaded}
        if path not in allowed and not any(path == root or path.startswith(root + os.sep) for root in roots)
    )
    if not loaded or outside:
        raise IdentityError(f"{role} worker loaded files outside the hashed identity: {outside or sorted(set(loaded))}")


def identity_digest(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
