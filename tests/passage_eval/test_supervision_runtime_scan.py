"""System package metadata must not weaken the scan of declared runtime artifacts."""

from dataclasses import replace
from pathlib import Path

import pytest

from passage_eval.supervision import environment, sandbox, treehash


def _layout(*roots: Path, venv_root: str | None = None) -> sandbox.RuntimeLayout:
    return sandbox.RuntimeLayout(
        python="/usr/bin/python3",
        real_executable="/usr/bin/python3",
        venv_root=venv_root,
        base_prefix="/usr",
        site_dirs=(),
        roots=tuple(map(str, roots)),
        pth_import_lines=0,
        editable_roots={},
        pth_roots=(),
    )


def _redirect_system_tree(monkeypatch, system: Path) -> None:
    # Keep the walk, hashing and cache real; avoid reading the host's whole /usr.
    hash_tree = treehash.hash_tree

    def redirected(path, **kwargs):
        if Path(path) == Path("/usr"):
            return replace(hash_tree(system, **kwargs), path="/usr")
        return hash_tree(path, **kwargs)

    monkeypatch.setattr(treehash, "hash_tree", redirected)


def test_system_json_metadata_is_hashed_but_declared_json_artifacts_stay_flagged(tmp_path, monkeypatch):
    system = tmp_path / "system"
    metadata = system / "local/share/vcpkg/versions/o-"
    metadata.mkdir(parents=True)
    (metadata / "onnxruntime.json").write_text('{"versions": []}\n')
    (metadata / "onnxruntime-gpu.json").write_text('{"versions": []}\n')
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "onnxruntime.json").write_text("{}\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "onnxruntime.json").write_text("{}\n")
    _redirect_system_tree(monkeypatch, system)
    cache = treehash.TreeHashCache(tmp_path / "cache")

    for source in ("computed", "stat_signature_cache"):
        bound = environment.build_bound_set(
            _layout(Path("/usr")),
            isolation="bwrap",
            declared=[candidate],
            extra_runtime=[runtime],
            protected=[],
            staging_base=tmp_path,
            cache=cache,
        )
        try:
            system_record = next(record for record in bound.environment if record.path == "/usr")
            assert system_record.files == 2 and system_record.hash_source == source
            found = {row["tree"]: row["entries"] for row in bound.forbidden_scan()["found"]}
            assert found == {str(candidate): ["onnxruntime.json"], str(runtime): ["onnxruntime.json"]}
        finally:
            bound.close()


@pytest.mark.parametrize("root, explicit_bind", [("/usr", True), ("/usr/local", True), ("/usr/local/venv", False)])
def test_declared_runtime_within_system_tree_keeps_the_strict_scan(tmp_path, monkeypatch, root, explicit_bind):
    system = tmp_path / "system"
    metadata = system / "local/share/vcpkg/versions/o-"
    metadata.mkdir(parents=True)
    (metadata / "onnxruntime.json").write_text('{"versions": []}\n')
    _redirect_system_tree(monkeypatch, system)
    bound = environment.build_bound_set(
        _layout(Path("/usr"), venv_root=None if explicit_bind else root),
        isolation="bwrap",
        declared=[],
        extra_runtime=[Path(root)] if explicit_bind else [],
        protected=[],
        staging_base=tmp_path,
    )
    try:
        assert bound.forbidden_scan()["clean"] is False
        assert bound.forbidden_scan()["found"][0]["entries"] == ["local/share/vcpkg/versions/o-/onnxruntime.json"]
    finally:
        bound.close()


@pytest.mark.parametrize(
    "artifact",
    [
        "lib/python3/site-packages/onnxruntime/__init__.py",
        "lib/python3/site-packages/onnxruntime_gpu-1.20.0.dist-info/METADATA",
        "lib/libonnxruntime.so.1.20.0",
        "lib/libonnxruntime_providers_shared.so",
        "lib/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so",
        "lib/onnxruntime.py",
        "share/onnxruntime-1.20.0.whl",
        "share/libonnxruntime.json",
    ],
)
def test_system_metadata_does_not_hide_runtime_packages_or_libraries(tmp_path, monkeypatch, artifact):
    system = tmp_path / "system"
    system.mkdir()
    (system / "onnxruntime.json").write_text('{"versions": []}\n')
    path = system / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"runtime artifact")
    _redirect_system_tree(monkeypatch, system)
    cache = treehash.TreeHashCache(tmp_path / "cache")

    for source in ("computed", "stat_signature_cache"):
        bound = environment.build_bound_set(
            _layout(Path("/usr")),
            isolation="bwrap",
            declared=[],
            protected=[],
            staging_base=tmp_path,
            cache=cache,
        )
        try:
            scan = bound.forbidden_scan()
            assert scan["clean"] is False
            assert len(scan["found"]) == 1
            assert scan["found"][0]["tree"] == "/usr"
            assert "onnxruntime.json" not in scan["found"][0]["entries"]
            assert bound.environment[0].hash_source == source
        finally:
            bound.close()
