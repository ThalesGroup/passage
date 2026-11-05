# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""What is bound into a candidate sandbox, and the identity of every byte of it.

A :class:`BoundSet` holds three kinds of trees:

``candidate_staged`` (``bwrap``)
    declared candidate files and directories, copied into a private content-addressed read-only
    area (hashed while copied) and bound from there at their original path. Nothing is bound
    that was not hashed, and the original cannot change what the sandbox sees after the copy.
``candidate_in_place`` (``process``)
    the same declared paths, hashed where they are. There is no mount namespace at this level, so
    there is nothing to bind; the digest is re-checked at the end of every series instead.
``environment``
    the interpreter environment that the sandbox binds (virtual environment, base prefix, editable
    source roots, extra ``--runtime-bind`` trees and, under ``bwrap``, ``/usr``). These trees are too
    large to copy, so they are hashed in place; a stat signature detects any change during the run.
    Their digests may be reused from an on-disk cache keyed by that signature; each record says
    whether its digest was computed or reused.

Declared candidate paths that overlap a protected path are refused before anything is copied.

A worker whose entry point the operator pinned (:meth:`BoundSet.pinned_view`) does not get the whole set: it gets the
interpreter environment and the one staged tree of its own package, so that the digest the operator pinned describes
every candidate-side byte that worker can import.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from passage_eval.supervision import sandbox, treehash
from passage_eval.supervision.identity import IdentityError, PathView
from passage_eval.supervision.treehash import StageArea, TreeHashCache, TreeRecord

#: Name prefixes (lowercase, base name of any entry) of ONNX Runtime distributions and shared libraries. A tree that
#: holds one is scanned by the supervisor while it is hashed; the rule does not depend on anything a worker reports.
FORBIDDEN_RUNTIME_PREFIXES = ("onnxruntime", "libonnxruntime")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def forbidden_runtime_name(name: str) -> bool:
    """True for ``onnxruntime*`` and ``libonnxruntime*`` (packages, ``*.dist-info`` and ``*.so`` files alike)."""
    return name.startswith(FORBIDDEN_RUNTIME_PREFIXES)


def _forbidden_system_runtime_name(name: str) -> bool:
    """Ignore ONNX Runtime JSON indexes in ``/usr``; declared artifact trees retain the strict rule."""
    if name.startswith("onnxruntime") and name.endswith(".json"):
        return False
    return forbidden_runtime_name(name)


@dataclass(frozen=True)
class RoleBinds:
    """What one worker role is given: its binds and the directories added to its ``sys.path``."""

    binds: tuple[sandbox.Bind, ...]
    import_paths: tuple[str, ...]
    narrowed: bool
    tree: str | None = None  # original path of the pinned package (a candidate tree, or a path in the environment)


def tree_cache_from(setting: str | None) -> TreeHashCache | None:
    """``None`` / ``''`` / ``'off'`` disable the cache; any other value is the cache directory."""
    if setting is None or setting in {"", "off"}:
        return None
    return TreeHashCache(setting)


def default_cache_directory() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "passage-eval", "tree-hash")


@dataclass
class BoundSet:
    isolation: str
    stage: StageArea | None
    candidate: list[TreeRecord] = field(default_factory=list)
    environment: list[TreeRecord] = field(default_factory=list)
    interpreter_aliases: list[sandbox.Bind] = field(default_factory=list)
    view: PathView = field(default_factory=PathView)
    os_layer: dict[str, Any] = field(default_factory=dict)

    # -- what the sandbox gets -------------------------------------------------------------

    @property
    def binds(self) -> list[sandbox.Bind]:
        if self.isolation != "bwrap":
            return []
        binds = [sandbox.Bind(record.path, record.path, "runtime") for record in self.environment]
        binds += [
            sandbox.Bind(
                record.staged_path or record.path, record.path, "candidate_staged" if record.staged_path else "runtime"
            )
            for record in self.candidate
        ]
        binds += self.interpreter_aliases
        return binds

    def candidate_roots(self) -> list[Path]:
        return [Path(record.path) for record in self.candidate]

    def pinned_view(self, package: str | Path) -> RoleBinds:
        """The binds of a worker whose entry point lives in ``package`` (a directory or a single module file).

        The role gets every interpreter-environment bind (virtual environment, evaluator sources, ``/usr``) and, when the
        package lies in a declared candidate tree, one bind of that package's staged copy at its original path, so that
        its parent directory holds nothing else. Sibling modules, the predictor, weights and every other declared path are
        absent from the sandbox and cannot be imported. ``import_paths`` is that parent directory (empty when the
        package lies in the environment, which is on ``sys.path`` anyway). Without a sandbox (``process``) nothing is
        bound and the file system is shared: the view is recorded as not enforced.
        """
        original = Path(os.path.realpath(package))
        runtime = tuple(bind for bind in self.binds if bind.role == "runtime")
        for record in self.candidate:
            root = Path(record.path)
            if not _inside(original, root):
                continue
            if self.isolation != "bwrap" or record.staged_path is None:
                return RoleBinds((), (str(original.parent),), narrowed=False, tree=str(original))
            source = Path(record.staged_path) / original.relative_to(root)
            return RoleBinds(
                (*runtime, sandbox.Bind(str(source), str(original), "candidate_staged")),
                (str(original.parent),),
                narrowed=True,
                tree=str(original),
            )
        return RoleBinds(
            runtime, (), narrowed=self.isolation == "bwrap", tree=str(original)
        )  # lives in the environment

    def forbidden_scan(self) -> dict[str, Any]:
        """What the supervisor found while walking the bound trees: ``onnxruntime*`` names (any role)."""
        trees = [*self.environment, *self.candidate]
        found = [
            {"tree": record.path, "role": record.role, "count": record.flagged, "entries": record.flagged_listed}
            for record in trees
            if record.flagged
        ]
        return {
            "rule": (
                f"base name starts with one of {list(FORBIDDEN_RUNTIME_PREFIXES)}; "
                "implicit system /usr excludes onnxruntime*.json metadata"
            ),
            "evidence": "supervisor_walk_of_the_hashed_trees",
            "trees_scanned": len(trees),
            "found": found,
            "clean": not found,
        }

    def covered_roots(self) -> list[Path]:
        return self.candidate_roots() + [Path(record.path) for record in self.environment]

    def is_covered(self, path: str | Path) -> bool:
        resolved = Path(os.path.realpath(path))
        return any(_inside(resolved, root) for root in self.covered_roots())

    # -- identity -----------------------------------------------------------------------

    @property
    def binding(self) -> str:
        return "staged_read_only_copy" if self.isolation == "bwrap" else "in_place_hash_rechecked"

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "binding": self.binding,
            "candidate": [record.to_manifest() for record in sorted(self.candidate, key=lambda r: r.path)],
            "environment": [record.to_manifest() for record in sorted(self.environment, key=lambda r: r.path)],
            "os_layer": dict(self.os_layer),
        }
        if self.interpreter_aliases:
            manifest["interpreter_aliases"] = [
                {"target": bind.target, "source": bind.source}
                for bind in sorted(self.interpreter_aliases, key=lambda b: b.target)
            ]
        return manifest

    def recheck(self) -> None:
        """Fail closed if any bound byte differs from what was hashed (called before every artifact)."""
        for record in self.candidate:
            if record.staged_path is not None:
                try:
                    self.stage.verify(record)
                except treehash.TreeError as error:
                    raise IdentityError(str(error)) from None
            else:
                again = treehash.hash_tree(
                    record.path, role=record.role, symlinks="refuse", strict=True, exclude_dirs=record.excluded_dirs
                )
                if again.sha256 != record.sha256:
                    raise IdentityError(f"declared candidate tree {record.path} changed during the run")
                if (
                    record.signature is not None
                    and treehash.current_signature(record.path, exclude_dirs=record.excluded_dirs) != record.signature
                ):
                    raise IdentityError(
                        f"declared candidate tree {record.path} was touched during the run (stat signature differs)"
                    )
        for record in self.environment:
            if treehash.current_signature(record.path) != record.signature:
                raise IdentityError(f"runtime tree {record.path} changed during the run")

    # -- extension -----------------------------------------------------------------------

    def with_declared(self, paths: Iterable[str | Path], protected: Sequence[str | Path]) -> "BoundSet":
        """A new set that also covers ``paths`` (declared files/directories not yet covered)."""
        new = replace(
            self,
            candidate=list(self.candidate),
            environment=list(self.environment),
            interpreter_aliases=list(self.interpreter_aliases),
            view=PathView(list(self.view.mapping)),
            os_layer=dict(self.os_layer),
        )
        stage_declared(new, paths, protected)
        return new

    def close(self) -> None:
        if self.stage is not None:
            self.stage.close()


def _declared_unique(paths: Iterable[str | Path]) -> list[Path]:
    """Resolved declared paths without duplicates and without paths nested inside another declared path."""
    resolved = sorted({Path(os.path.realpath(p)) for p in paths}, key=lambda p: (len(p.parts), str(p)))
    kept: list[Path] = []
    for path in resolved:
        if not any(_inside(path, known) for known in kept):
            kept.append(path)
    return kept


def stage_declared(bound: BoundSet, paths: Iterable[str | Path], protected: Sequence[str | Path]) -> list[TreeRecord]:
    """Stage (``bwrap``) or hash in place (``process``) the declared paths that are not covered yet."""
    originals = []
    for path in paths:
        if not os.path.lexists(path):
            raise sandbox.SandboxError(f"declared bind does not exist: {path}")
        originals.append(Path(os.path.realpath(path)))
    sandbox.check_binds(originals, protected)  # before anything is copied
    added = []
    for path in _declared_unique(originals):
        if any(_inside(path, root) for root in bound.candidate_roots()):
            continue  # already staged as part of a larger tree
        if any(_inside(root, path) for root in bound.candidate_roots()):
            raise IdentityError(
                f"declared path {path} contains an already staged tree; declare the larger "
                f"path together with the others"
            )
        if bound.isolation == "bwrap":
            if bound.stage is None:
                raise IdentityError("a bwrap run needs a staging area")
            try:
                record = bound.stage.stage(path)
            except treehash.TreeError as error:
                raise IdentityError(str(error)) from None
            bound.view.add(path, record.staged_path)
        else:
            try:
                record = treehash.hash_tree(
                    path,
                    role="candidate_in_place",
                    symlinks="refuse",
                    strict=True,
                    exclude_dirs=treehash.CANDIDATE_EXCLUDES,
                    flag=forbidden_runtime_name,
                )
            except treehash.TreeError as error:
                raise IdentityError(str(error)) from None
        bound.candidate.append(record)
        added.append(record)
    return added


def build_bound_set(
    layout: sandbox.RuntimeLayout,
    *,
    isolation: str,
    declared: Iterable[str | Path],
    protected: Sequence[str | Path],
    extra_runtime: Iterable[str | Path] = (),
    cache: TreeHashCache | None = None,
    staging_base: str | Path | None = None,
) -> BoundSet:
    """Stage the declared candidate paths and hash the interpreter environment."""
    sys.dont_write_bytecode = True  # the supervisor must not add bytecode files to trees it has just hashed
    bound = BoundSet(
        isolation=isolation,
        stage=StageArea(staging_base, flag=forbidden_runtime_name) if isolation == "bwrap" else None,
    )
    try:
        stage_declared(bound, declared, protected)
        explicit_runtime_roots = [Path(os.path.realpath(r)) for r in extra_runtime]
        runtime_roots = [Path(r) for r in layout.roots] + explicit_runtime_roots
        # Deduplication folds nested runtimes into /usr. Keep that whole tree strict
        # when it covers a declared runtime or a separate interpreter environment.
        strict_roots = explicit_runtime_roots + ([Path(layout.venv_root)] if layout.venv_root is not None else [])
        strict_system_runtime = any(_inside(root, Path("/usr")) for root in strict_roots) or any(
            root != Path("/usr") and _inside(root, Path("/usr")) for root in runtime_roots
        )
        unresolved = [(Path(path), reason) for path, reason in layout.needs_declaration]
        if isolation != "bwrap":
            runtime_roots = [r for r in runtime_roots if r != Path("/usr")]
            runtime_roots += [path for path, _ in unresolved]  # nothing is bound at this level: hash what is named
            bound.os_layer = {
                "path": "/usr",
                "hashed": False,
                "reason": "process isolation binds nothing and does not confine the file system",
            }
        else:
            covering = bound.candidate_roots() + runtime_roots
            missing = [
                (path, reason) for path, reason in unresolved if not any(_inside(path, root) for root in covering)
            ]
            if missing:
                raise sandbox.SandboxError(
                    "the interpreter environment names locations that are not bound on its say-so (a candidate's .pth "
                    "files and pyvenv.cfg could otherwise expose any directory): "
                    + "; ".join(f"{path} ({reason})" for path, reason in missing[:6])
                    + ". Declare each one that is meant to be visible with --runtime-bind (environment) or --ro-bind "
                    "(candidate files)."
                )
        sandbox.check_binds(runtime_roots, protected)
        kept = []
        for root in _declared_unique(runtime_roots):
            if any(_inside(root, candidate) for candidate in bound.candidate_roots()):
                continue  # served by the staged copy of a declared path
            kept.append(root)
        for root in kept:
            record = treehash.hash_tree(
                root,
                role="environment",
                symlinks="record",
                cache=cache,
                flag=(
                    _forbidden_system_runtime_name
                    if root == Path("/usr") and not strict_system_runtime
                    else forbidden_runtime_name
                ),
            )
            bound.environment.append(record)
        if isolation == "bwrap":
            mounted = [Path(record.path) for record in [*bound.environment, *bound.candidate]]
            reserved = [Path(path) for path in ("/usr", "/proc", "/dev", "/tmp", sandbox.SANDBOX_HOME)]
            for target_text, source_text in layout.executable_aliases:
                target, source = Path(target_text), Path(source_text)
                if not any(_inside(source, Path(record.path)) for record in bound.environment):
                    raise sandbox.SandboxError(f"interpreter alias target is not in a hashed runtime tree: {source}")
                if any(target == path or target in path.parents or path in target.parents for path in mounted):
                    raise sandbox.SandboxError(f"interpreter alias overlaps a mounted tree: {target}")
                if any(target == path or target in path.parents for path in reserved) or any(
                    path in target.parents for path in reserved if path != Path("/tmp")
                ):
                    raise sandbox.SandboxError(f"interpreter alias overlaps a reserved sandbox path: {target}")
                if any(
                    target == Path(path) or target in Path(path).parents or Path(path) in target.parents
                    for path in protected
                ):
                    raise sandbox.SandboxError(f"interpreter alias overlaps a protected path: {target}")
                bound.interpreter_aliases.append(sandbox.Bind(str(source), str(target), "runtime", "symlink"))
            usr = next((r for r in bound.environment if r.path == "/usr"), None)
            bound.os_layer = {
                "path": "/usr",
                "hashed": usr is not None,
                "sha256": None if usr is None else usr.sha256,
                "hash_source": None if usr is None else usr.hash_source,
            }
    except BaseException:
        bound.close()
        raise
    return bound
