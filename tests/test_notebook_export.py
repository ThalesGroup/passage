"""Publication staging and notebook receipts never certify file existence alone."""

import json
import os
import shutil
from pathlib import Path

import nbformat
import pytest

from passage.notebook_export import prepare_snapshot_notebooks, stage_snapshot, write_staged_bytes

RESOLUTIONS = (64, 128, 256, 512, 1024, 2048, 4096)


def _sources(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    for name in ("dataset", "display", "metrics", "passage_eval_metrics"):
        nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell("resolution = 256\noutput_dir = None", metadata={"tags": ["parameters"]}),
                nbformat.v4.new_code_cell("report = {}\nsection_statuses = {}"),
            ]
        )
        nbformat.write(nb, root / f"_{name}.ipynb")
    nested = root / "passage-eval"
    nested.mkdir()
    nbformat.write(nb, nested / "walkthrough.ipynb")
    return root


def test_complete_matrix_and_recursive_inventory_are_prepared_without_execution(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    stage = tmp_path / "stage"
    notebooks = _sources(tmp_path)
    manifest = prepare_snapshot_notebooks(source, stage, notebooks_dirpath=notebooks)
    matrix = [entry for entry in manifest["notebooks"] if entry["group"] == "resolution"]
    assert len(matrix) == 28
    assert {(entry["resolution"], Path(entry["destination"]).name) for entry in matrix} == {
        (resolution, f"{name}.ipynb")
        for resolution in RESOLUTIONS
        for name in ("dataset", "display", "metrics", "passage_eval_metrics")
    }
    assert (stage / "notebooks/passage-eval/walkthrough.ipynb").is_file()
    assert (stage / "notebooks/_dataset.ipynb").is_file()
    assert all(entry["execution"]["status"] == "not_run" for entry in matrix)
    learned = next(
        e for e in matrix if e["resolution"] == 256 and e["destination"].endswith("passage_eval_metrics.ipynb")
    )
    assert learned["sections"]["learned_s2_s4"]["required"] is True
    assert learned["sections"]["learned_s2_s4"]["status"] == "not_run"
    assert "checkpoint" in learned["sections"]["learned_s2_s4"]["reason"]


def test_parseable_or_tampered_output_is_not_reused_and_changed_input_invalidates(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    shard = source / "64x64/validation/data.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"first")
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    calls = []

    def execute(input_path, output_path, parameters, **kwargs):
        calls.append(output_path)
        nb = nbformat.read(input_path, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                cell.execution_count = i + 1
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="checked")]
        nbformat.write(nb, output_path)
        artifacts = Path(parameters["output_dir"])
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "_export_receipt.json").write_text(json.dumps({"report": {}, "section_statuses": {}}))
        (artifacts / f"{artifacts.name}-{parameters['resolution']}.json").write_text("{}")

    monkeypatch.setattr("passage.notebook_export.execute_notebook", execute)
    args = dict(notebooks_dirpath=notebooks, resolutions=[64], execute=True)
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 4
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 4
    output = stage / "64x64/display.ipynb"
    output.write_text(json.dumps(nbformat.v4.new_notebook()))
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 5
    shard.write_bytes(b"changed")
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 9
    (stage / "notebook-artifacts/64x64/metrics/_export_receipt.json").unlink()
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 10


def _executed_outputs(input_path, output_path, parameters, *, omitted=None, timestamp_ns=1_000_000_000, **kwargs):
    notebook = nbformat.read(input_path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            cell.execution_count = index + 1
    nbformat.write(notebook, output_path)
    artifacts = Path(parameters["output_dir"])
    files = {
        "_export_receipt.json": json.dumps({"report": {"figures": ["figure.pdf", "figure.png"]}}),
        f"{artifacts.name}-{parameters['resolution']}.json": "{}",
        "figure.pdf": "new pdf",
        "figure.png": "new png",
    }
    for name, content in files.items():
        if name != omitted:
            path = artifacts / name
            path.write_text(content)
            os.utime(path, ns=(timestamp_ns, timestamp_ns))


def test_fresh_outputs_with_coarse_timestamps_preserve_unwritten_artifacts(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    artifacts = stage / "notebook-artifacts/64x64/dataset"
    notes = artifacts / "notes"
    notes.mkdir(parents=True)
    (notes / "keep.txt").write_text("user notes")
    (notes / "empty").mkdir()
    (notes / "link").symlink_to("keep.txt")
    (artifacts / "figure.png").write_text("old png")
    args = dict(notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=_executed_outputs)

    manifest = prepare_snapshot_notebooks(source, stage, **args)
    # Identical bytes and timestamps on a forced retry still come from this run.
    prepare_snapshot_notebooks(source, stage, force=True, **args)

    assert manifest["notebooks"][0]["execution"]["status"] == "executed"
    assert (artifacts / "figure.png").read_text() == "new png"
    assert (artifacts / "figure.png").stat().st_mtime_ns == 1_000_000_000
    assert (notes / "keep.txt").read_text() == "user notes"
    assert (notes / "empty").is_dir()
    assert (notes / "link").readlink() == Path("keep.txt")
    assert "notebook-artifacts/64x64/dataset/notes/keep.txt" not in manifest["notebooks"][0]["artifacts"]


@pytest.mark.parametrize("omitted", ["_export_receipt.json", "dataset-64.json", "figure.png"])
def test_unchanged_old_output_cannot_satisfy_new_execution(tmp_path, omitted):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    artifacts = stage / "notebook-artifacts/64x64/dataset"
    artifacts.mkdir(parents=True)
    old = artifacts / omitted
    old.write_text("old reviewed output")
    # Even a misleading future timestamp cannot qualify an unwritten artifact.
    future_ns = 4_000_000_000_000_000_000
    os.utime(old, ns=(future_ns, future_ns))

    def execute(input_path, output_path, parameters, **kwargs):
        skipped = omitted if Path(parameters["output_dir"]).name == "dataset" else None
        _executed_outputs(input_path, output_path, parameters, omitted=skipped, timestamp_ns=future_ns)

    with pytest.raises(ValueError, match="Missing fresh required output|Missing or external declared figure"):
        prepare_snapshot_notebooks(
            source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=execute
        )

    assert old.read_text() == "old reviewed output"
    assert old.stat().st_mtime_ns == future_ns
    assert sorted(path.name for path in artifacts.iterdir()) == [omitted]
    manifest = json.loads((stage / "notebook-export-manifest.json").read_text())
    assert manifest["notebooks"][0]["execution"]["status"] == "failed"


def test_retention_never_overwrites_new_files_or_follows_directory_links(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    artifacts = stage / "notebook-artifacts/64x64/dataset"
    notes = artifacts / "notes"
    notes.mkdir(parents=True)
    (notes / "keep.txt").write_text("user notes")
    (notes / "refresh.txt").write_text("old notes")
    (artifacts / "dangling").write_text("old file")
    (artifacts / "redirect").mkdir()
    (artifacts / "redirect/keep.txt").write_text("old nested file")
    outside = tmp_path / "outside"
    outside.mkdir()

    def execute(input_path, output_path, parameters, **kwargs):
        _executed_outputs(input_path, output_path, parameters)
        root = Path(parameters["output_dir"])
        (root / "notes").mkdir()
        (root / "notes/refresh.txt").write_text("new notes")
        (root / "dangling").symlink_to("missing")
        (root / "redirect").symlink_to(outside, target_is_directory=True)

    prepare_snapshot_notebooks(
        source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=execute
    )

    assert (notes / "keep.txt").read_text() == "user notes"
    assert (notes / "refresh.txt").read_text() == "new notes"
    assert (artifacts / "dangling").readlink() == Path("missing")
    assert (artifacts / "redirect").readlink() == outside
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("root_state", ["missing", "file", "symlink"])
def test_failed_execution_restores_reviewed_bundle_after_root_replacement(tmp_path, root_state):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    artifacts = stage / "notebook-artifacts/64x64/dataset"
    artifacts.mkdir(parents=True)
    (artifacts / "keep.txt").write_text("reviewed artifact")
    output = stage / "64x64/dataset.ipynb"
    output.parent.mkdir(parents=True)
    output.write_text("reviewed notebook")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("external file")

    def execute(*args, **kwargs):
        shutil.rmtree(artifacts)
        if root_state == "file":
            artifacts.write_text("partial file")
        elif root_state == "symlink":
            artifacts.symlink_to(outside, target_is_directory=True)
        raise RuntimeError("artifact root replaced")

    with pytest.raises(RuntimeError, match="artifact root replaced"):
        prepare_snapshot_notebooks(
            source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=execute
        )

    assert (artifacts / "keep.txt").read_text() == "reviewed artifact"
    assert output.read_text() == "reviewed notebook"
    assert (outside / "keep.txt").read_text() == "external file"


@pytest.mark.parametrize("linked_component", ["notebook-artifacts", "notebook-artifacts/64x64/dataset"])
def test_symlinked_artifact_directory_is_rejected_without_touching_target(tmp_path, linked_component):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("external file")
    link = stage / linked_component
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact directory must not be a symlink"):
        prepare_snapshot_notebooks(
            source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=_executed_outputs
        )

    assert (outside / "keep.txt").read_text() == "external file"
    assert sorted(path.name for path in outside.iterdir()) == ["keep.txt"]


def test_readonly_hardlinks_and_atomic_replacement_preserve_source(tmp_path):
    source = tmp_path / "dataset"
    shard = source / "64x64/train/data.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"original")
    shard.chmod(0o444)
    stat = shard.stat()
    stage = tmp_path / "stage"
    manifest = stage_snapshot(source, stage, resolutions=[64])
    staged = stage / "64x64/train/data.parquet"
    assert staged.stat().st_ino == stat.st_ino
    assert manifest["files"][0]["method"] == "hardlink"
    assert manifest["content_verified"] is False
    write_staged_bytes(staged, b"replacement")
    assert staged.stat().st_ino != stat.st_ino
    assert shard.read_bytes() == b"original"
    assert shard.stat().st_mode == stat.st_mode


def test_writable_source_requires_reflink_and_never_silently_copies(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    shard = source / "64x64/train/data.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"original")
    monkeypatch.setattr("passage.notebook_export._reflink", lambda *args: False)
    with pytest.raises(OSError, match="reflink"):
        stage_snapshot(source, tmp_path / "stage", resolutions=[64])
    assert shard.stat().st_mode & 0o200


def test_verified_hash_mismatch_and_source_overlap_are_rejected(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "README.md").write_text("original")
    with pytest.raises(ValueError, match="separate"):
        stage_snapshot(source, source / "publication")
    with pytest.raises(ValueError, match="SHA256"):
        stage_snapshot(source, tmp_path / "stage", expected_hashes={"README.md": "0" * 64}, revision="pinned")


def test_errors_and_unexecuted_cells_block_receipts_without_replacing_good_output(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    output = stage / "64x64/dataset.ipynb"
    output.parent.mkdir(parents=True)
    output.write_text("previous reviewed output")

    def execute(input_path, output_path, parameters, **kwargs):
        nbformat.write(nbformat.read(input_path, as_version=4), output_path)

    monkeypatch.setattr("passage.notebook_export.execute_notebook", execute)
    with pytest.raises(ValueError, match="unexecuted"):
        prepare_snapshot_notebooks(source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True)
    assert output.read_text() == "previous reviewed output"
    manifest = json.loads((stage / "notebook-export-manifest.json").read_text())
    assert manifest["notebooks"][0]["execution"]["status"] == "failed"


def test_execute_uses_current_interpreter_offline_and_finite_timeout(tmp_path, monkeypatch):
    from passage.notebook_export import execute_notebook

    calls = {}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.update(command=command, kwargs=kwargs)

        def communicate(self, timeout):
            calls["timeout"] = timeout
            return b"", None

    monkeypatch.setattr("passage.notebook_export.subprocess.Popen", Process)
    execute_notebook(tmp_path / "in.ipynb", tmp_path / "out.ipynb", {}, cwd=tmp_path, cell_timeout=7, job_timeout=11)
    assert calls["timeout"] == 11
    assert calls["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert "--execution-timeout" in calls["command"]
    assert calls["command"][1:3] == ["-m", "papermill"]
    assert calls["kwargs"]["start_new_session"] is True


def test_public_notebook_parameters_are_portable_with_original_execution_recorded(tmp_path, monkeypatch):
    source = tmp_path / "private-source"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"

    def execute(input_path, output_path, parameters, **kwargs):
        nb = nbformat.read(input_path, as_version=4)
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                cell.execution_count = index + 1
                cell.outputs = []
        nbformat.write(nb, output_path)
        root = Path(parameters["output_dir"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "_export_receipt.json").write_text('{"report": {}, "section_statuses": {}}')
        (root / f"{root.name}-{parameters['resolution']}.json").write_text("{}")

    monkeypatch.setattr("passage.notebook_export.execute_notebook", execute)
    manifest = prepare_snapshot_notebooks(
        source, stage, notebooks_dirpath=notebooks, resolutions=[64], revision="a" * 40, execute=True
    )
    notebook = nbformat.read(stage / "64x64/dataset.ipynb", as_version=4)
    params = notebook.metadata.passage.published_parameters
    assert params["project_path"] is None
    assert params["dataset_path"] is None
    assert not Path(params["output_dir"]).is_absolute()
    assert notebook.metadata.passage.execution_parameters["dataset_path"] == str(source)
    assert manifest["notebooks"][0]["execution"]["portable_parameter_edits"]


def test_vision_stage_requires_registry_and_consumes_registered_results(tmp_path, monkeypatch):
    from passage.notebook_export import vision_study_report

    assert vision_study_report("02_segmentation_backbone_screen", tmp_path)["status"] == "not_run"
    stage = tmp_path / "02_segmentation_backbone_screen"
    stage.mkdir()
    (stage / "study_registry.json").write_text(
        json.dumps({"stage": stage.name, "procedure_ids": ["procedure"], "dataset_revision": "pinned"})
    )

    def report(*args):
        return {
            "status": "admitted",
            "learned_status": "admitted",
            "records": [{"procedure_id": "procedure"}],
            "receipts": [{"procedure_id": "procedure", "dataset": {"revision": "pinned"}, "learned": True}],
            "task_ids": ["real-id"],
        }

    monkeypatch.setattr("passage.notebook_reports.evaluation_report", report)
    actual = vision_study_report(stage.name, tmp_path)
    assert actual["status"] == "admitted"
    assert actual["task_ids"] == ["real-id"]
    (stage / "study_registry.json").write_text(
        json.dumps({"stage": stage.name, "procedure_ids": ["other"], "dataset_revision": "pinned"})
    )
    assert vision_study_report(stage.name, tmp_path)["status"] == "not_run"


def test_failed_retry_restores_reviewed_artifacts_and_notebook(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    stage = tmp_path / "stage"
    notebooks = _sources(tmp_path)
    output = stage / "64x64/dataset.ipynb"
    output.parent.mkdir(parents=True)
    output.write_text("reviewed notebook")
    artifacts = stage / "notebook-artifacts/64x64/dataset"
    artifacts.mkdir(parents=True)
    good_files = {
        "_export_receipt.json": b"original receipt",
        "dataset-64.json": b"original report",
        "figure.png": b"original figure",
    }
    for name, data in good_files.items():
        (artifacts / name).write_bytes(data)

    def execute(*args, **kwargs):
        (artifacts / "dataset-64.json").write_bytes(b"failed partial output")
        (artifacts / "figure.png").write_bytes(b"failed partial figure")
        raise RuntimeError("finite execution failure")

    monkeypatch.setattr("passage.notebook_export.execute_notebook", execute)
    with pytest.raises(RuntimeError, match="finite execution failure"):
        prepare_snapshot_notebooks(source, stage, notebooks_dirpath=notebooks, resolutions=[64], execute=True)
    assert output.read_text() == "reviewed notebook"
    for name, data in good_files.items():
        assert (artifacts / name).read_bytes() == data


def test_job_timeout_terminates_process_group(tmp_path, monkeypatch):
    import signal
    import subprocess
    from passage.notebook_export import execute_notebook

    calls = {"communicate": 0}

    class Process:
        pid = 123456
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout=None):
            calls["communicate"] += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("papermill", timeout)
            return b"", None

    monkeypatch.setattr("passage.notebook_export.subprocess.Popen", Process)
    monkeypatch.setattr("passage.notebook_export.os.killpg", lambda pid, sig: calls.update(pid=pid, signal=sig))
    with pytest.raises(TimeoutError, match="exceeded"):
        execute_notebook(
            tmp_path / "in.ipynb", tmp_path / "out.ipynb", {}, cwd=tmp_path, cell_timeout=7, job_timeout=11
        )
    assert calls == {"communicate": 2, "pid": 123456, "signal": signal.SIGKILL}


def test_complete_pinned_sha256_inventory_verifies_source(tmp_path):
    import hashlib

    source = tmp_path / "dataset"
    source.mkdir()
    (source / "README.md").write_bytes(b"identity")
    expected = {"README.md": hashlib.sha256(b"identity").hexdigest()}
    manifest = stage_snapshot(source, tmp_path / "stage", revision="a" * 40, expected_hashes=expected)
    assert manifest["content_verified"] is True
    assert manifest["files"][0]["pinned_hash_verified"] is True


def test_prepare_without_execution_preserves_existing_publication_notebook(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    output = stage / "64x64/display.ipynb"
    output.parent.mkdir(parents=True)
    output.write_text("reviewed publication notebook")
    manifest = prepare_snapshot_notebooks(source, stage, notebooks_dirpath=notebooks, resolutions=[64])
    assert output.read_text() == "reviewed publication notebook"
    entry = next(e for e in manifest["notebooks"] if e["destination"].endswith("display.ipynb"))
    assert entry["execution"]["status"] == "not_run"
    assert entry["retained_existing_output"] is True


def test_source_output_stripping_does_not_strip_executed_global_reports(tmp_path):
    from passage.export import _strip_analysis_notebook_outputs

    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "1 + 1", execution_count=1, outputs=[nbformat.v4.new_output("stream", name="stdout", text="2")]
            )
        ]
    )
    sources = tmp_path / "notebooks/passage-eval"
    sources.mkdir(parents=True)
    nbformat.write(nb, sources / "walkthrough.ipynb")
    reports = tmp_path / "notebooks/paper"
    reports.mkdir()
    nbformat.write(nb, reports / "dataset_statistics.ipynb")
    stripped, scanned = _strip_analysis_notebook_outputs(tmp_path)
    assert stripped == scanned == 1
    assert nbformat.read(reports / "dataset_statistics.ipynb", as_version=4).cells[0].execution_count == 1


def _qualification_authentication_fixture(root):
    import hashlib

    pin = "c" * 40
    manifest = {
        "schema_version": 1,
        "root": "/frozen/private/source",
        "repo": "fixture/passage",
        "revision": pin,
        "repo_type": "dataset",
        "files": [{"path": "a", "size": 10, "algorithm": "sha256", "expected_hash": "b" * 64}],
    }
    manifest_id = hashlib.sha256(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "root": manifest["root"],
        "repo": manifest["repo"],
        "revision": pin,
        "manifest_sha256": manifest_id,
        "status": "complete",
        "content_verified": True,
        "required_files": 1,
        "required_bytes": 10,
        "verified_files": 1,
        "verified_bytes": 10,
        **dict.fromkeys(
            ("mismatched_files", "missing_files", "changed_files", "error_files", "interrupted_files", "pending_files"),
            0,
        ),
    }
    observed_stat = {"device": 1, "inode": 2, "size": 10, "mtime_ns": 3, "ctime_ns": 4}
    progress = {
        "path": "a",
        "status": "verified",
        "expected_hash": "b" * 64,
        "observed_hash": "b" * 64,
        "manifest_sha256": manifest_id,
        "stat": observed_stat,
        "stat_after": observed_stat,
    }
    paths = {
        "source-authentication/manifest.json": manifest,
        "source-authentication/receipt.json": receipt,
        "source-authentication/progress.jsonl": progress,
    }
    for name, value in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
    return paths


def _executed_export_fixture(input_path, output_path, parameters, **kwargs):
    notebook = nbformat.read(input_path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            cell.execution_count = index + 1
            cell.outputs = []
    nbformat.write(notebook, output_path)
    artifacts = Path(parameters["output_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "_export_receipt.json").write_text('{"report": {}, "section_statuses": {}}')
    (artifacts / f"{artifacts.name}-{parameters['resolution']}.json").write_text("{}")


@pytest.mark.parametrize("declaration", ["complete", {"status": "complete", "reason": "Validated producer receipts"}])
def test_complete_producer_sections_are_admitted_without_changing_receipt(tmp_path, declaration):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    paper = tmp_path / "paper-notebooks"
    paper.mkdir()
    nbformat.write(nbformat.read(notebooks / "_dataset.ipynb", as_version=4), paper / "dataset_statistics.ipynb")
    receipt = {"report": {}, "section_statuses": {"qualification": declaration, "qualified_references": declaration}}

    def execute(input_path, output_path, parameters, **kwargs):
        _executed_export_fixture(input_path, output_path, parameters, **kwargs)
        Path(parameters["output_dir"], "_export_receipt.json").write_text(json.dumps(receipt))

    manifest = prepare_snapshot_notebooks(
        source,
        tmp_path / "stage",
        notebooks_dirpath=notebooks,
        paper_notebooks_dirpath=paper,
        resolutions=[64],
        execute=True,
        executor=execute,
    )
    for destination, required in [
        ("64x64/dataset.ipynb", True),
        ("64x64/metrics.ipynb", False),
        ("notebooks/paper/dataset_statistics.ipynb", False),
    ]:
        entry = next(e for e in manifest["notebooks"] if e["destination"] == destination)
        assert entry["execution"]["status"] == "executed"
        assert entry["sections"]["qualification"]["status"] == "admitted"
        assert entry["sections"]["qualification"]["required"] is required
        assert entry["sections"]["qualified_references"]["status"] == "admitted"
        assert entry["sections"]["qualified_references"]["required"] is False
        if isinstance(declaration, dict):
            assert entry["sections"]["qualification"]["reason"] == "Validated producer receipts"
        assert entry["producer_receipt"] == receipt


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("required", "required"),
        ("admitted", "admitted"),
        ("not_run", "not_run"),
        ("unsupported", "unsupported"),
        ("pending", "not_run"),
        ("unexpected_success", "not_run"),
    ],
)
def test_producer_section_statuses_preserve_admission_and_pending_semantics(tmp_path, status, expected):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)

    def execute(input_path, output_path, parameters, **kwargs):
        _executed_export_fixture(input_path, output_path, parameters, **kwargs)
        Path(parameters["output_dir"], "_export_receipt.json").write_text(
            json.dumps({"report": {}, "section_statuses": {"qualification": status}})
        )

    manifest = prepare_snapshot_notebooks(
        source, tmp_path / "stage", notebooks_dirpath=notebooks, resolutions=[64], execute=True, executor=execute
    )
    entry = next(e for e in manifest["notebooks"] if e["destination"] == "64x64/dataset.ipynb")
    assert entry["sections"]["qualification"]["status"] == expected
    assert entry["sections"]["qualification"]["required"] is True
    assert entry["producer_receipt"]["section_statuses"]["qualification"] == status


def test_complete_section_does_not_override_pending_learned_results(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)

    def execute(input_path, output_path, parameters, **kwargs):
        _executed_export_fixture(input_path, output_path, parameters, **kwargs)
        Path(parameters["output_dir"], "_export_receipt.json").write_text(
            json.dumps({"report": {"learned_status": "not_run"}, "section_statuses": {"learned_S2_S4": "complete"}})
        )

    manifest = prepare_snapshot_notebooks(
        source, tmp_path / "stage", notebooks_dirpath=notebooks, resolutions=[256], execute=True, executor=execute
    )
    entry = next(e for e in manifest["notebooks"] if e["destination"] == "256x256/passage_eval_metrics.ipynb")
    assert entry["sections"]["learned_s2_s4"]["status"] == "not_run"
    assert entry["sections"]["learned_s2_s4"]["required"] is True


def test_qualification_export_copies_exact_compact_evidence_and_preserves_historical_binding(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    qualification = tmp_path / "qualification"
    files = _qualification_authentication_fixture(qualification)
    finding = qualification / "route-audit/findings/task.json"
    finding.parent.mkdir(parents=True)
    finding.write_text('{"root": "/frozen/private/source", "scope": "pending finding"}\n')
    for name in (
        "metadata/census.sqlite",
        "route-audit/cache.db",
        "metadata/raster.tif",
        "source-authentication/huge.parquet",
        "route-audit/extra.json",
    ):
        path = qualification / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be copied")
    stage = tmp_path / "stage"
    manifest = prepare_snapshot_notebooks(
        source, stage, notebooks_dirpath=notebooks, resolutions=[64], qualification_root=qualification
    )
    bundle = stage / "notebook-artifacts/qualification-evidence"
    expected = {*files, "route-audit/findings/task.json"}
    assert {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} == expected
    for name in expected:
        assert (bundle / name).read_bytes() == (qualification / name).read_bytes()
    assert set(manifest["auxiliary"]["qualification"]) == expected
    assert manifest["qualification_evidence"]["completion"]["source_authentication"] is True
    assert manifest["qualification_evidence"]["completion"]["full_campaign"] is False
    assert manifest["qualification_evidence"]["source"]["root"] == "/frozen/private/source"
    assert manifest["content_verified"] is False
    assert manifest["input_verification_scope"] == "direct_source_fingerprints"
    assert all(e["execution"]["status"] == "not_run" for e in manifest["notebooks"])
    prepared = nbformat.read(stage / "64x64/dataset.ipynb", as_version=4)
    assert (
        prepared.metadata.passage.published_parameters["qualification_root"]
        == "notebook-artifacts/qualification-evidence"
    )
    assert prepared.metadata.passage.execution_parameters is None
    assert "qualification_root = 'notebook-artifacts/qualification-evidence'" in prepared.cells[0].source


def test_qualification_bundle_changes_invalidate_reuse_and_published_parameter_is_portable(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    qualification = tmp_path / "qualification"
    _qualification_authentication_fixture(qualification)
    stage = tmp_path / "stage"
    calls = []

    def execute(input_path, output_path, parameters, **kwargs):
        calls.append(parameters.copy())
        _executed_export_fixture(input_path, output_path, parameters, **kwargs)

    args = dict(
        notebooks_dirpath=notebooks, resolutions=[64], qualification_root=qualification, execute=True, executor=execute
    )
    manifest = prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 4
    bundle = stage / "notebook-artifacts/qualification-evidence"
    assert calls[0]["qualification_root"] == str(bundle)
    published = nbformat.read(stage / "64x64/dataset.ipynb", as_version=4)
    assert (
        published.metadata.passage.published_parameters["qualification_root"]
        == "notebook-artifacts/qualification-evidence"
    )
    assert published.metadata.passage.execution_parameters["qualification_root"] == str(bundle)
    prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 4
    progress = qualification / "source-authentication/progress.jsonl"
    progress.write_bytes(progress.read_bytes() * 2)
    updated = prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 8
    assert (
        updated["auxiliary"]["qualification"]["source-authentication/progress.jsonl"]["sha256"]
        != manifest["auxiliary"]["qualification"]["source-authentication/progress.jsonl"]["sha256"]
    )
    assert updated["qualification_evidence"]["completion"]["source_authentication"] is True


def test_qualification_missing_bundle_is_pending_and_changed_bundle_drops_stale_files(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    qualification = tmp_path / "qualification"
    _qualification_authentication_fixture(qualification)
    stage = tmp_path / "stage"
    args = dict(notebooks_dirpath=notebooks, resolutions=[64], qualification_root=qualification)
    prepare_snapshot_notebooks(source, stage, **args)
    progress = qualification / "source-authentication/progress.jsonl"
    progress.unlink()
    updated = prepare_snapshot_notebooks(source, stage, **args)
    assert not (stage / "notebook-artifacts/qualification-evidence/source-authentication/progress.jsonl").exists()
    assert updated["qualification_evidence"]["completion"]["source_authentication"] is False
    assert "source-authentication/progress.jsonl" in updated["qualification_evidence"]["missing_prerequisites"]
    absent = prepare_snapshot_notebooks(
        source,
        tmp_path / "absent-stage",
        notebooks_dirpath=notebooks,
        resolutions=[64],
        qualification_root=tmp_path / "missing",
    )
    assert absent["qualification_evidence"]["status"] == "not_run"
    assert absent["auxiliary"]["qualification"] == {}


def test_invalid_qualification_snapshot_does_not_replace_previous_portable_evidence(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    notebooks = _sources(tmp_path)
    qualification = tmp_path / "qualification"
    _qualification_authentication_fixture(qualification)
    stage = tmp_path / "stage"
    args = dict(notebooks_dirpath=notebooks, resolutions=[64], qualification_root=qualification)
    prepare_snapshot_notebooks(source, stage, **args)
    bundle = stage / "notebook-artifacts/qualification-evidence"
    reviewed = {p.relative_to(bundle).as_posix(): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    receipt_path = qualification / "source-authentication/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest_sha256"] = "c" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="manifest hash"):
        prepare_snapshot_notebooks(source, stage, **args)
    assert reviewed == {p.relative_to(bundle).as_posix(): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}


def test_execution_reuse_requires_content_hash_when_size_and_mtime_match(tmp_path, monkeypatch):
    """A same-length data change must invalidate previously executed notebook receipts."""
    import os

    source = tmp_path / "dataset"
    shard = source / "64x64/train/data.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"first")
    initial = shard.stat()
    notebooks = _sources(tmp_path)
    stage = tmp_path / "stage"
    calls = []

    def execute(input_path, output_path, parameters, **kwargs):
        calls.append(output_path)
        notebook = nbformat.read(input_path, as_version=4)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                cell.execution_count = i + 1
                cell.outputs = []
        nbformat.write(notebook, output_path)
        artifacts = Path(parameters["output_dir"])
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "_export_receipt.json").write_text('{"report":{},"section_statuses":{}}')
        (artifacts / f"{artifacts.name}-{parameters['resolution']}.json").write_text("{}")

    monkeypatch.setattr("passage.notebook_export.execute_notebook", execute)
    args = dict(notebooks_dirpath=notebooks, resolutions=[64], execute=True)
    first = prepare_snapshot_notebooks(source, stage, **args)
    shard.write_bytes(b"other")
    os.utime(shard, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    second = prepare_snapshot_notebooks(source, stage, **args)
    assert len(calls) == 8
    assert (
        first["inputs"]["64x64/train/data.parquet"]["sha256"] != second["inputs"]["64x64/train/data.parquet"]["sha256"]
    )


def test_publication_admission_rejects_missing_required_template(tmp_path):
    from passage.notebook_export import validate_publication_notebooks

    manifest = {"notebooks": [], "execution_requested": True}
    with pytest.raises(ValueError, match="required publication notebook"):
        validate_publication_notebooks(manifest, tmp_path, resolutions=[64], require_global_reports=False)


def test_publication_admission_requires_all_twenty_eight_resolution_outputs(tmp_path):
    from passage.notebook_export import validate_publication_notebooks

    manifest = {
        "notebooks": [
            {"destination": f"{resolution}x{resolution}/{kind}.ipynb", "execution": {"status": "executed"}}
            for resolution in RESOLUTIONS
            for kind in ("dataset", "display", "metrics", "passage_eval_metrics")
            if (resolution, kind) != (4096, "display")
        ]
    }
    with pytest.raises(ValueError, match="4096x4096/display.ipynb"):
        validate_publication_notebooks(manifest, tmp_path, require_global_reports=False)
