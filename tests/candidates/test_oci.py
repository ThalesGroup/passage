"""OCI host boundaries: input-only binds, archive safety and process cleanup."""

import io
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
import pytest

from passage_eval.campaign.candidate import oci


def _tar(path, entries):
    with tarfile.open(path, "w") as archive:
        for name, kind, value in entries:
            item = tarfile.TarInfo(name)
            if kind == "link":
                item.type = tarfile.SYMTYPE
                item.linkname = value
                archive.addfile(item)
            else:
                data = value.encode()
                item.size = len(data)
                item.mode = 0o755
                archive.addfile(item, io.BytesIO(data))


@pytest.mark.parametrize(
    "entries",
    [
        [("../escaped", "file", "bad")],
        [("alias", "link", "/tmp"), ("alias/escaped", "file", "bad")],
    ],
)
def test_archive_rejects_escape_or_symlink_ancestor(tmp_path, entries):
    archive = tmp_path / "root.tar"
    _tar(archive, entries)
    with pytest.raises(ValueError):
        oci._extract_rootfs(archive, tmp_path / "rootfs")
    assert not (tmp_path / "escaped").exists()


def test_archive_preserves_absolute_symlink_without_following_it(tmp_path):
    archive = tmp_path / "root.tar"
    _tar(archive, [("usr/bin/python", "file", "binary"), ("bin", "link", "/usr/bin")])
    root = tmp_path / "rootfs"
    oci._extract_rootfs(archive, root)
    assert os.readlink(root / "bin") == "/usr/bin"
    before, records = oci.tree_manifest(root)
    assert any(item["type"] == "symlink" for item in records)
    (root / "usr/bin/python").write_text("changed")
    assert before != oci.tree_manifest(root)[0]


def test_spec_mounts_only_input_file_and_no_systemd_resources(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    inputs = tmp_path / "labels-parent/scenes.npz"
    inputs.parent.mkdir()
    inputs.touch()
    output = tmp_path / "output"
    output.mkdir()
    spec = oci._build_spec(
        tmp_path / "rootfs", package, inputs, output, python="/usr/local/bin/python", env=[], device="cpu"
    )
    mounts = {item["destination"]: item for item in spec["mounts"]}
    assert mounts["/inputs/scenes.npz"]["source"] == str(inputs)
    assert "/inputs" not in mounts
    assert "ro" in mounts["/inputs/scenes.npz"]["options"]
    assert "ro" in mounts["/candidate"]["options"]
    assert "rw" in mounts["/output"]["options"]
    assert spec["root"]["readonly"] is True
    assert all(not value for value in spec["process"]["capabilities"].values())
    assert spec["process"]["noNewPrivileges"] is True
    assert {"user", "pid", "mount", "network", "cgroup"} <= {n["type"] for n in spec["linux"]["namespaces"]}
    assert "resources" not in spec["linux"] and "cgroupsPath" not in spec["linux"]
    assert "passage_eval.campaign.candidate.worker" in spec["process"]["args"]


def test_input_archive_rejects_targets_even_in_a_file(tmp_path):
    path = tmp_path / "scene.npz"
    np.savez(path, terrain=[1], sample_ids=["a"], obstacles=[False], starts=[[0, 0]], goals=[[0, 1]], cost_to_go=[2])
    with pytest.raises(ValueError, match="input-only"):
        oci._check_inputs(path)


def test_execute_timeout_kills_child_group_and_drains(tmp_path):
    child = tmp_path / "child.py"
    marker = tmp_path / "escaped"
    child.write_text(
        'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'
        + repr(f"import time;from pathlib import Path;time.sleep(1);Path({str(marker)!r}).touch()")
        + "])\ntime.sleep(20)\n"
    )
    result = oci._execute([sys.executable, str(child)], tmp_path / "stdout", tmp_path / "stderr", timeout_seconds=0.15)
    assert result["timed_out"] and result["process_reaped"] and result["returncode"] != 0
    time.sleep(1.1)
    assert not marker.exists()


def test_prepare_always_removes_its_created_container_on_export_failure(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    [{"Id": "sha256:" + "a" * 64, "Config": {"Env": []}, "Architecture": "arm64", "Os": "linux"}]
                ),
                "",
            )
        if args[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(args, 0, "container-id\n", "")
        if args[:2] == ["docker", "export"]:
            raise subprocess.CalledProcessError(1, args, stderr="export failed")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(oci.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        oci.prepare_rootfs("local/example", tmp_path / "prepared")
    assert any(call[:2] == ["docker", "rm"] and call[-1].startswith("passage-pilot-") for call in calls)


def test_cdi_rejects_unapproved_uuid_before_reading_spec(tmp_path):
    with pytest.raises(ValueError, match="approved GPU"):
        oci.prepare_cdi("GPU-other", tmp_path / "missing.yaml")


def test_cdi_only_selected_gpu_common_edits_and_validates_char_nodes(tmp_path):
    library = tmp_path / "libcuda.so.1"
    library.write_bytes(b"library")
    path = tmp_path / "cdi.yaml"
    spec = {
        "kind": "nvidia.com/gpu",
        "devices": [
            {
                "name": oci.ALLOWED_GPU,
                "containerEdits": {"deviceNodes": [{"path": "/dev/null", "major": 1, "minor": 3}]},
            },
            {"name": "GPU-other", "containerEdits": {"deviceNodes": [{"path": "/dev/zero"}]}},
        ],
        "containerEdits": {
            "mounts": [{"hostPath": str(library), "containerPath": "/usr/lib/libcuda.so.1"}],
            "hooks": [
                {
                    "hookName": "createContainer",
                    "path": "/usr/bin/unused",
                    "args": ["unused", "create-symlinks", "--link", "libcuda.so.1::/usr/lib/libcuda.so"],
                }
            ],
        },
    }
    import yaml

    path.write_text(yaml.safe_dump(spec))
    plan = oci.prepare_cdi(oci.ALLOWED_GPU, path)
    assert [node["path"] for node in plan["device_nodes"]] == ["/dev/null"]
    assert plan["symlinks"] == [{"link": "/usr/lib/libcuda.so", "target": "libcuda.so.1"}]
    assert plan["dependencies"][0]["sha256"] == oci.file_sha256(library)
    spec["devices"][0]["containerEdits"]["deviceNodes"][0]["path"] = str(library)
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="character device"):
        oci.prepare_cdi(oci.ALLOWED_GPU, path)


def test_container_path_interprets_absolute_links_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "usr/lib").mkdir(parents=True)
    (root / "lib").symlink_to("/usr/lib")
    assert oci._container_path(root, "/lib/driver.so") == root / "usr/lib/driver.so"
    (root / "cycle").symlink_to("/cycle")
    with pytest.raises(ValueError, match="cycle"):
        oci._container_path(root, "/cycle/driver.so")


def test_cdi_alias_mounts_resolve_chains_without_host_hooks(tmp_path):
    library = tmp_path / "driver"
    library.write_bytes(b"driver")
    plan = {
        "gpu_uuid": oci.ALLOWED_GPU,
        "mounts": [{"containerPath": "/usr/lib/versioned.so", "hostPath": str(library)}],
        "device_nodes": [],
        "env": [],
        "symlinks": [
            {"link": "/usr/lib/libcuda.so", "target": "libcuda.so.1"},
            {"link": "/usr/lib/libcuda.so.1", "target": "versioned.so"},
        ],
    }
    spec = {"process": {"env": []}, "mounts": []}
    oci._apply_cdi(spec, plan)
    mounts = {item["destination"]: item for item in spec["mounts"]}
    assert mounts["/usr/lib/libcuda.so"]["source"] == str(library)
    assert "ro" in mounts["/usr/lib/libcuda.so"]["options"]
    assert "hooks" not in spec
    assert f"CUDA_VISIBLE_DEVICES={oci.ALLOWED_GPU}" in spec["process"]["env"]


def test_runtime_copy_supports_output_on_another_filesystem(tmp_path, monkeypatch):
    import errno

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_bytes(b"sealed bytes")
    source.chmod(0o755)

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(oci.os, "link", cross_device)
    oci._copy_root_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode == source.stat().st_mode


@pytest.mark.parametrize("failed_action", ["kill", "delete", "list"])
def test_cleanup_attempts_every_step_after_command_timeout(tmp_path, monkeypatch, failed_action):
    calls = []

    def fake_run(args, **kwargs):
        action = next(value for value in args if value in {"kill", "delete", "list"})
        calls.append(action)
        if action == failed_action:
            raise subprocess.TimeoutExpired(args, 10)
        return subprocess.CompletedProcess(args, 0, "[]" if action == "list" else "", "")

    monkeypatch.setattr(oci.subprocess, "run", fake_run)
    cleanup = []
    result = oci._cleanup_runtime(["runc", "--root", str(tmp_path / "state")], "passage-pilot-fault", tmp_path, cleanup)
    assert calls == ["kill", "delete", "list"]
    assert any(
        item["action"] == {"kill": "kill_all", "delete": "delete_force", "list": "inspect_state"}[failed_action]
        and item.get("error_type") == "TimeoutExpired"
        for item in cleanup
    )
    assert result["cleanup_errors"]
    if failed_action == "list":
        assert result["child_namespace_drained"] is None
        assert result["private_runtime_state_empty"] is None
    else:
        assert result["child_namespace_drained"] is True


def _library_plan(tmp_path):
    library = tmp_path / "libdriver.so.1"
    library.write_bytes(b"original driver")
    plan = {
        "dependencies": [oci._dependency(library)],
        "mounts": [{"hostPath": str(library), "containerPath": "/usr/lib/libdriver.so.1"}],
        "device_nodes": [],
        "env": [],
        "gpu_uuid": oci.ALLOWED_GPU,
        "symlinks": [{"link": "/usr/lib/libdriver.so", "target": "libdriver.so.1"}],
    }
    return library, plan


def test_cdi_replacement_after_planning_is_rejected_before_snapshot(tmp_path):
    library, plan = _library_plan(tmp_path)
    library.write_bytes(b"new unsealed driver")
    with pytest.raises(ValueError, match="CDI dependency changed"):
        oci._snapshot_cdi(plan, tmp_path / "snapshot")


def test_cdi_mounts_and_aliases_use_independent_snapshot_bytes(tmp_path):
    library, plan = _library_plan(tmp_path)
    snapshot = oci._snapshot_cdi(plan, tmp_path / "snapshot")
    library.write_bytes(b"host replacement after snapshot")
    spec = {"process": {"env": []}, "mounts": []}
    oci._apply_cdi(spec, snapshot)
    mounts = {item["destination"]: item for item in spec["mounts"]}
    bound = Path(mounts["/usr/lib/libdriver.so.1"]["source"])
    assert bound != library and bound.read_bytes() == b"original driver"
    assert mounts["/usr/lib/libdriver.so"]["source"] == str(bound)
    assert oci._verify_cdi_snapshot(snapshot)["regular_files"] == 1
    bound.write_bytes(b"mutated private snapshot")
    with pytest.raises(ValueError, match="CDI snapshot changed"):
        oci._verify_cdi_snapshot(snapshot)


def test_run_preserves_execution_failure_and_still_deletes_and_inspects(tmp_path, monkeypatch):
    from .test_package import package_fixture

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    manifest, _ = package_fixture(package_dir)
    root = tmp_path / "rootfs"
    (root / "usr/bin").mkdir(parents=True)
    (root / "usr/bin/python").write_text("fixture only")
    archive = tmp_path / "rootfs.tar"
    archive.write_bytes(b"archive fixture")
    receipt = tmp_path / "rootfs-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": oci.ROOTFS_SCHEMA,
                "rootfs": str(root),
                "rootfs_tree_sha256": oci.tree_manifest(root)[0],
                "archive": str(archive),
                "archive_sha256": oci.file_sha256(archive),
                "image_config_metadata": {"Env": ["PATH=/usr/bin"]},
                "image_inspect_id": "fixture",
            }
        )
    )
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        sample_ids=["a"],
        terrain=np.zeros((1, 256, 256)),
        obstacles=np.zeros((1, 256, 256), bool),
        starts=[[0, 0]],
        goals=[[0, 1]],
    )
    calls = []

    def fake_run(args, **kwargs):
        action = next(value for value in args if value in {"kill", "delete", "list"})
        calls.append(action)
        if action == "kill":
            raise subprocess.TimeoutExpired(args, 10)
        return subprocess.CompletedProcess(args, 0, "[]" if action == "list" else "", "")

    def fail_execution(*args, **kwargs):
        raise RuntimeError("original execution failure")

    monkeypatch.setattr(oci.subprocess, "run", fake_run)
    monkeypatch.setattr(oci, "_execute", fail_execution)
    with pytest.raises(RuntimeError, match="original execution failure"):
        oci.run_oci(receipt, manifest, inputs, tmp_path / "output")
    assert calls == ["kill", "delete", "list"]
    failure = json.loads(next(tmp_path.glob(".passage-oci-*/failure.json")).read_text())
    assert failure["error"] == "RuntimeError: original execution failure"
    assert failure["result"]["cleanup_errors"][0]["error_type"] == "TimeoutExpired"


def test_cdi_hash_uses_the_same_bytes_as_yaml_parse(tmp_path, monkeypatch):
    import yaml

    path = tmp_path / "cdi.yaml"
    original = yaml.safe_dump(
        {"kind": "nvidia.com/gpu", "devices": [{"name": oci.ALLOWED_GPU, "containerEdits": {}}]}
    ).encode()
    path.write_bytes(original)
    original_load = yaml.safe_load

    def replace_file_after_parse(data):
        parsed = original_load(data)
        path.write_text("replacement after read")
        return parsed

    monkeypatch.setattr(yaml, "safe_load", replace_file_after_parse)
    plan = oci.prepare_cdi(oci.ALLOWED_GPU, path)
    assert plan["cdi_sha256"] == oci.sha256(original).hexdigest()
    assert plan["cdi_sha256"] != oci.file_sha256(path)


def test_cdi_character_identity_is_rechecked_without_opening_devices(tmp_path):
    plan = {
        "dependencies": [],
        "device_nodes": [
            {
                "path": "/dev/null",
                "hostPath": "/dev/null",
                "major": 1,
                "minor": 3,
                "mode": stat.S_IMODE(Path("/dev/null").stat().st_mode),
            }
        ],
    }
    assert oci._verify_cdi_snapshot(plan)["character_devices"] == 1
    plan["device_nodes"][0]["minor"] = 5
    with pytest.raises(ValueError, match="character device identity changed"):
        oci._verify_cdi_snapshot(plan)


def test_cleanup_accepts_runc_null_for_an_empty_instance_list(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "null\n" if "list" in args else "", "")

    monkeypatch.setattr(oci.subprocess, "run", fake_run)
    result = oci._cleanup_runtime(["runc"], "passage-pilot-empty", tmp_path, [])
    assert result["private_runtime_state_empty"] is True
    assert result["child_namespace_drained"] is True
    assert not result["cleanup_errors"]


@pytest.mark.parametrize("state_json", ["{}", "0", '""', "{invalid"])
def test_cleanup_reports_unknown_for_unusable_state_json(tmp_path, monkeypatch, state_json):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, state_json if "list" in args else "", "")

    monkeypatch.setattr(oci.subprocess, "run", fake_run)
    result = oci._cleanup_runtime(["runc"], "passage-pilot-invalid-state", tmp_path, [])
    assert result["private_runtime_state_empty"] is None
    assert result["child_namespace_drained"] is None
    assert result["cleanup_errors"]
