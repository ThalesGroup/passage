"""Safe local publication staging and explicit notebook execution receipts.

Staging never changes the source inode. Writable inputs need reflink support;
ordinary copying is an explicit, capacity-checked opt-in. Hashes without pinned
expected identities remain provisional, even when every notebook executes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nbformat

RESOLUTIONS = (64, 128, 256, 512, 1024, 2048, 4096)
TEMPLATES = ("dataset", "display", "metrics", "passage_eval_metrics")
MANIFEST_NAME = "notebook-export-manifest.json"
PAPER_REPORTS = (
    "cost_model_path_divergence",
    "dataset_qualification",
    "dataset_statistics",
    "obstacles_showcase",
    "solver_runtime_scaling",
)
VISION_REPORTS = (
    "01_resource_calibration_and_resolution",
    "02_segmentation_backbone_screen",
    "03_segmentation_loss_ablation",
    "04_regression_backbone_screen",
    "05_regression_loss_ablation",
    "06_formulation_showdown",
    "07_cost_model_and_obstacle_transfer",
    "08_inference_latency_and_qualitative",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, *, content: bool = True) -> dict[str, Any]:
    st = path.stat()
    return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": _hash(path) if content else None}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_staged_bytes(path: str | Path, content: bytes) -> None:
    """Atomically replace a publication file with a new inode, including hardlinks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    write_staged_bytes(path, (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode())


def _separate_roots(source: Path, staging: Path) -> None:
    if source == staging or source in staging.parents or staging in source.parents:
        raise ValueError("Source and publication staging must be separate trees")


def _selected(path: Path, resolutions: Sequence[int] | None) -> bool:
    first = path.parts[0]
    if "x" not in first or not first.split("x")[0].isdigit():
        return True
    return resolutions is None or int(first.split("x")[0]) in resolutions


def _reflink(source: Path, destination: Path) -> bool:
    """Try Linux FICLONE without changing source permissions or contents."""
    try:
        import fcntl

        with source.open("rb") as src, destination.open("wb") as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
        return True
    except OSError:
        destination.unlink(missing_ok=True)
        return False


def stage_snapshot(
    source_root: str | Path,
    staging_root: str | Path,
    *,
    resolutions: Sequence[int] | None = None,
    revision: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    allow_copy: bool = False,
    hash_inputs: bool = False,
) -> dict[str, Any]:
    """Stage unchanged Parquets read-only; other files always use new inodes.

    Pinned ``expected_hashes`` are SHA256 content/LFS identities keyed by relative
    path. Only a complete matching inventory plus a revision verifies content.
    Default Parquet fingerprints use size/mtime, avoiding a hidden multi-TB scan.
    ``allow_copy`` is explicit and checks free space before copying writable data.
    """
    source, staging = Path(source_root).resolve(), Path(staging_root).resolve()
    _separate_roots(source, staging)
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = [p for p in sorted(source.rglob("*")) if p.is_file() and _selected(p.relative_to(source), resolutions)]
    expected = dict(expected_hashes or {})
    for path in paths:
        rel = path.relative_to(source).as_posix()
        if rel in expected and _hash(path) != expected[rel]:
            raise ValueError(f"SHA256 mismatch against pinned source identity: {rel}")
    missing = set(expected) - {p.relative_to(source).as_posix() for p in paths}
    if missing:
        raise ValueError(f"Expected source files are missing: {sorted(missing)}")
    staging.mkdir(parents=True, exist_ok=True)
    if allow_copy:
        required = sum(p.stat().st_size for p in paths)
        if shutil.disk_usage(staging).free < required:
            raise OSError(f"Insufficient staging space for explicit copy: {required} bytes required")
    records = []
    for path in paths:
        rel = path.relative_to(source)
        destination = staging / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Never follow a staged symlink into the source snapshot.
        if destination.parent.resolve() != destination.parent:
            raise ValueError(f"Staged directory must not be a symlink: {destination.parent}")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        method = "copy"
        try:
            if path.suffix == ".parquet":
                readonly = not (path.stat().st_mode & 0o222)
                if readonly and path.stat().st_dev == staging.stat().st_dev:
                    temporary.unlink()
                    os.link(path, temporary)
                    method = "hardlink"
                elif _reflink(path, temporary):
                    method = "reflink"
                    temporary.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
                elif allow_copy:
                    shutil.copyfile(path, temporary)
                    temporary.chmod(0o444)
                else:
                    raise OSError(
                        f"Read-only staging requires reflink support for writable source {rel}; "
                        "ordinary copy requires explicit allow_copy=True"
                    )
            else:
                shutil.copyfile(path, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        fingerprint = _fingerprint(path, content=hash_inputs or rel.as_posix() in expected or path.suffix != ".parquet")
        records.append(
            {
                "path": rel.as_posix(),
                **fingerprint,
                "method": method,
                "pinned_hash_verified": rel.as_posix() in expected,
            }
        )
    verified = bool(revision and records and all(r["pinned_hash_verified"] for r in records))
    manifest = {
        "schema_version": 1,
        "source_root": str(source),
        "staging_root": str(staging),
        "source_revision": revision,
        "content_verified": verified,
        "verification_status": "verified" if verified else "unverified",
        "files": records,
    }
    _write_json(staging / "snapshot-staging-manifest.json", manifest)
    return manifest


def execute_notebook(
    input_path: Path,
    output_path: Path,
    parameters: dict[str, Any],
    *,
    cwd: Path,
    cell_timeout: int,
    job_timeout: int,
    offline: bool = True,
) -> None:
    """Run the current frozen interpreter, with cell and process-tree deadlines."""
    if cell_timeout <= 0 or job_timeout <= 0:
        raise ValueError("Notebook cell and job timeouts must be positive")
    environment = os.environ.copy()
    if offline:
        environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", UV_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="passage-notebook-") as directory:
        params = Path(directory) / "parameters.json"
        params.write_text(json.dumps(parameters))
        command = [
            sys.executable,
            "-m",
            "papermill",
            str(input_path),
            str(output_path),
            "-k",
            "python3",
            "--execution-timeout",
            str(cell_timeout),
            "-f",
            str(params),
        ]
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        try:
            output, _ = process.communicate(timeout=job_timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError(f"Notebook job exceeded {job_timeout}s: {input_path.name}") from exc
        except BaseException:
            # Cancellation must stop the independent kernel/process session too.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(
                f"Notebook execution failed: {input_path.name}\n{output.decode(errors='replace')[-8000:]}"
            )


def _execution_evidence(path: Path) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    codes = [(i, c) for i, c in enumerate(notebook.cells) if c.cell_type == "code" and c.source.strip()]
    if not codes or any(c.execution_count is None for _, c in codes):
        raise ValueError(f"Notebook has unexecuted code cells: {path}")
    if any(o.output_type == "error" for _, c in codes for o in c.get("outputs", [])):
        raise ValueError(f"Notebook contains error outputs: {path}")
    return {
        "status": "executed",
        "notebook_sha256": _hash(path),
        "executed_cells": [
            {
                "cell_index": i,
                "execution_count": c.execution_count,
                "source_sha256": _canonical(c.source),
                "outputs_sha256": _canonical(c.get("outputs", [])),
                "output_count": len(c.get("outputs", [])),
            }
            for i, c in codes
        ],
    }


def validate_publication_notebooks(
    manifest: Mapping[str, Any],
    staging_root: str | Path,
    *,
    resolutions: Sequence[int] | None = None,
    require_global_reports: bool = True,
) -> dict[str, Any]:
    """Admit a fixed notebook inventory only with current execution and artifact hashes.

    Learned sections may remain explicitly not_run. Missing sources, outputs or
    execution evidence block publication even when the available jobs succeeded.
    """
    selected = tuple(RESOLUTIONS if resolutions is None else resolutions)
    if not selected or any(r not in RESOLUTIONS for r in selected):
        raise ValueError("Select supported PASSAGE resolutions")
    expected = {f"{r}x{r}/{kind}.ipynb" for r in selected for kind in TEMPLATES}
    if require_global_reports:
        expected.update(f"notebooks/paper/{name}.ipynb" for name in PAPER_REPORTS)
        expected.update(f"notebooks/vision/{name}.ipynb" for name in VISION_REPORTS)
    entries = manifest.get("notebooks", [])
    destinations = [entry.get("destination") for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate required publication notebook identity")
    indexed = {entry["destination"]: entry for entry in entries}
    missing = sorted(expected - indexed.keys())
    if missing:
        raise ValueError(f"Missing required publication notebook: {missing[0]}")
    root = Path(staging_root).resolve()
    verified = {}
    for destination in sorted(expected):
        entry = indexed[destination]
        if not entry.get("source_sha256") or not entry.get("identity_sha256"):
            raise ValueError(f"Missing source/identity for required publication notebook: {destination}")
        if entry.get("execution", {}).get("status") != "executed":
            raise ValueError(f"Unexecuted required publication notebook: {destination}")
        path = (root / destination).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Missing required publication notebook output: {destination}")
        evidence = _execution_evidence(path)
        if evidence["notebook_sha256"] != entry["execution"].get("notebook_sha256"):
            raise ValueError(f"Required publication notebook hash mismatch: {destination}")
        required = entry.get("required_outputs", [])
        artifacts = entry.get("artifacts", {})
        if destination not in required or not artifacts or len(required) < 3:
            raise ValueError(f"Incomplete required publication notebook artifacts: {destination}")
        for output in required:
            output_path = (root / output).resolve()
            expected_hash = evidence["notebook_sha256"] if output == destination else artifacts.get(output)
            if (
                root not in output_path.parents
                or not output_path.is_file()
                or not expected_hash
                or _hash(output_path) != expected_hash
            ):
                raise ValueError(f"Required publication notebook artifact hash mismatch: {output}")
        verified[destination] = evidence["notebook_sha256"]
    return {
        "status": "admitted",
        "required_notebooks": len(expected),
        "resolutions": list(selected),
        "global_reports_required": require_global_reports,
        "notebook_sha256": verified,
        "learned_results_required": False,
    }


def _sections(kind: str, resolution: int) -> dict[str, Any]:
    names = {
        "dataset": ["core_census", "qualification", "classical_diagnostics"],
        "display": ["real_tensor_and_routes"],
        "metrics": ["real_tensor_and_routes", "classical_diagnostics"],
        "passage_eval_metrics": ["classical_diagnostics", "learned_s2_s4"],
    }.get(kind, ["declared_results"])
    return {
        name: {
            "required": name != "learned_s2_s4" or resolution == 256,
            "status": "not_run",
            "reason": (
                "Requires admitted checkpoint and matched per-task S2–S4 result/timing receipts"
                if name == "learned_s2_s4"
                else "Notebook has not executed with declared inputs"
            ),
        }
        for name in names
    }


def vision_study_report(stage: str, result_root: str | Path | None, resolution: int = 256) -> dict[str, Any]:
    """Consume qualified stage results only when a declared registry binds them.

    ``result_root/<stage>/study_registry.json`` declares the exact ``stage``,
    ``procedure_ids`` and ``dataset_revision``. Standard evaluator receipts in
    that directory supply request, path, scoring and checkpoint hashes. The
    registry does not replace those checks or authorize training/measurement.
    """
    from passage.notebook_reports import evaluation_report

    report = {
        "stage": stage,
        "status": "not_run",
        "records": [],
        "task_ids": [],
        "reason": "Requires stage registry, declared data/targets, checkpoints and measured result receipts",
    }
    if result_root is None:
        return report
    directory = Path(result_root) / stage
    registry_path = directory / "study_registry.json"
    if not registry_path.is_file():
        report["reason"] = f"Missing stage registry: {stage}/study_registry.json"
        return report
    registry = json.loads(registry_path.read_text())
    report["registry_sha256"] = _hash(registry_path)
    report["registry"] = registry
    if registry.get("stage") != stage or not registry.get("procedure_ids") or not registry.get("dataset_revision"):
        report["reason"] = "Registry must bind stage, procedure identities and dataset revision"
        return report
    measured = evaluation_report(directory, resolution)
    receipts = measured.get("receipts", [])
    matched = bool(receipts) and all(
        r.get("procedure_id") in registry["procedure_ids"]
        and r.get("dataset", {}).get("revision") == registry["dataset_revision"]
        for r in receipts
    )
    learned_required = not stage.startswith("01_")
    if (
        not matched
        or measured.get("status") != "admitted"
        or (learned_required and measured.get("learned_records_status", measured.get("learned_status")) != "admitted")
    ):
        report["reason"] = "Stage-matched qualified results/checkpoints are missing or disagree with the registry"
        report["evaluation_prerequisites"] = measured
        return report
    return {
        **measured,
        **report,
        "status": "admitted",
        "records": measured["records"],
        "task_ids": measured.get("task_ids", []),
        "evaluation_receipts": receipts,
        "reason": "Stage registry matched hash-bound measured evaluator receipts; no launch cells ran",
    }


def _vision_report(source: Path) -> Any:
    """Export a report-only consumer; keep launch workflows in the source inventory."""
    return nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        cells=[
            nbformat.v4.new_markdown_cell(
                f"# {source.stem}\n\nReport for the declared campaign stage. "
                "Training and measurement require an explicit source workflow. Missing evidence remains pending."
            ),
            nbformat.v4.new_code_cell(
                "resolution = 256\ndataset_path = None\ndataset_repo = 'thalesgroup/passage'\n"
                "dataset_revision = None\noutput_dir = 'outputs/notebook-reports'\nresult_root = None",
                metadata={"tags": ["parameters"]},
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nimport json\n"
                "from passage.notebook_export import vision_study_report\n"
                "report = vision_study_report(" + repr(source.stem) + ", result_root, resolution)\n"
                "section_statuses = {'declared_results': {'status': report['status'], 'reason': report['reason']}}\n"
                "report['section_statuses'] = section_statuses\n"
                "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
                "Path(output_dir, f'{report[\"stage\"]}-{resolution}.json').write_text(json.dumps(report, indent=2))\n"
                "print(json.dumps(report, indent=2))"
            ),
        ],
    )


QUALIFICATION_BUNDLE_ROOT = Path("notebook-artifacts/qualification-evidence")
QUALIFICATION_FILES = (
    "source-authentication/manifest.json",
    "source-authentication/receipt.json",
    "source-authentication/progress.jsonl",
    "metadata/protocol.json",
    "metadata/summary.json",
    "metadata/audit_roster.json",
    "route-audit/protocol.json",
    "route-audit/summary.json",
)


def _stage_qualification_evidence(input_root: str | Path | None, staging: Path):
    """Copy compact, hash-bound receipts without changing their producer fields.

    Database, raster and dataset files are deliberately outside this inventory.
    Qualification completion is validated by the shared consumer, independently
    of this exporter's provisional fingerprints of current source Parquets.
    """
    if input_root is None:
        return None, {}, {"status": "not_run", "reason": "Qualification bundle was not supplied"}
    from passage.qualification_reports import qualification_report

    source = Path(input_root).resolve()
    destination = staging / QUALIFICATION_BUNDLE_ROOT
    if source.exists() and not source.is_dir():
        raise ValueError("qualification_root must name an evidence directory")
    names = [name for name in QUALIFICATION_FILES if (source / name).is_file()]
    names += [
        path.relative_to(source).as_posix()
        for path in sorted((source / "route-audit/findings").glob("*.json"))
        if path.is_file()
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination:
        raise ValueError("Qualification publication bundle must not be a symlink")

    def inspect(root):
        report = qualification_report(root)
        files = {name: {"sha256": _hash(root / name), "size_bytes": (root / name).stat().st_size} for name in names}
        evidence = {
            "bundle_root": QUALIFICATION_BUNDLE_ROOT.as_posix(),
            "status": report["status"],
            "source": report["source"],
            "completion": report["completion"],
            "binding": report["binding"],
            "missing_prerequisites": report["missing_prerequisites"],
            "interpretation": "Validated historical producer receipts; current input fingerprints are tracked separately",
            "files": files,
        }
        return files, evidence

    if source == destination:
        files, evidence = inspect(source)
        return destination, files, evidence
    with tempfile.TemporaryDirectory(prefix="qualification-bundle-", dir=destination.parent) as directory:
        prepared = Path(directory) / "prepared"
        prepared.mkdir()
        for name in names:
            path = prepared / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        files, evidence = inspect(prepared)
        backup = Path(directory) / "previous"
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(prepared, destination)
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise
    return destination, files, evidence


def _publication_parameters(parameters: dict[str, Any], relative: Path, revision: str | None) -> dict[str, Any]:
    portable = {
        **parameters,
        "dataset_path": None if revision else "dataset",
        "project_path": None,
        "output_dir": str(Path("outputs/notebook-reports") / relative.with_suffix("")),
        "result_root": None,
        "qualification_root": QUALIFICATION_BUNDLE_ROOT.as_posix() if parameters.get("qualification_root") else None,
    }
    for name in ("CONFIG_FILEPATH", "CALIBRATE_FILEPATH", "DATASET_DIRPATH"):
        portable.pop(name, None)
    return portable


def _portable_notebook(notebook, parameters: dict[str, Any], relative: Path, revision: str | None):
    """Record actual execution, then explicitly mark portable parameter edits."""
    portable = _publication_parameters(parameters, relative, revision)
    assignments = "\n".join(f"{name} = {value!r}" for name, value in portable.items()) + "\n"
    edits = []
    for index, cell in enumerate(notebook.cells):
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.cell_type == "code" and ("parameters" in tags or "injected-parameters" in tags):
            executed_hash = _canonical(cell.source)
            cell.source = ((cell.source + "\n") if "parameters" in tags else "") + assignments
            cell.metadata["passage_parameter_edit_after_execution"] = True
            edits.append(
                {
                    "cell_index": index,
                    "executed_source_sha256": executed_hash,
                    "published_source_sha256": _canonical(cell.source),
                }
            )
    if not edits:
        # Unparameterized fixture/missing-prerequisite sources still expose a replay cell.
        notebook.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                assignments, metadata={"tags": ["parameters"], "passage_parameter_edit_after_execution": True}
            ),
        )
        edits.append(
            {"cell_index": 0, "executed_source_sha256": None, "published_source_sha256": _canonical(assignments)}
        )
    notebook.metadata["passage"] = {
        "execution_parameters": parameters,
        "published_parameters": portable,
        "parameter_edits_after_execution": edits,
        "reproduction_note": "Portable parameter cells were edited after execution; outputs used execution_parameters. "
        "Set result_root to published stage bundles to reproduce learned results; absent bundles remain pending.",
    }
    return notebook, edits


def prepare_snapshot_notebooks(
    source_root: str | Path | None,
    staging_root: str | Path,
    *,
    notebooks_dirpath: str | Path = "notebooks",
    resolutions: Sequence[int] | None = None,
    repo_id: str = "thalesgroup/passage",
    revision: str | None = None,
    execute: bool = False,
    cell_timeout: int = 300,
    job_timeout: int = 1800,
    config_filepath: str | Path | None = None,
    style_filepath: str | Path | None = None,
    result_root: str | Path | None = None,
    qualification_root: str | Path | None = None,
    checkpoint_paths: Sequence[str | Path] | None = None,
    paper_notebooks_dirpath: str | Path | None = None,
    vision_notebooks_dirpath: str | Path | None = None,
    force: bool = False,
    hash_inputs: bool = False,
    parameters: Mapping[str, Any] | None = None,
    executor=None,
    _allow_prepared_root: bool = False,
) -> dict[str, Any]:
    """Prepare the 4×resolution matrix, recursive sources and global study reports.

    Execution is explicit. Reuse requires unchanged inputs, source/style/config,
    parameters, checkpoint/results, executed code and output hashes. Absence of
    authenticated input identities always remains visible in the receipt.
    """
    staging = Path(staging_root).resolve()
    source = Path(source_root).resolve() if source_root is not None else None
    if source is not None:
        if not (_allow_prepared_root and source == staging):
            _separate_roots(source, staging)
    elif not revision:
        raise ValueError("Explicit HF mode requires a pinned revision")
    if cell_timeout <= 0 or job_timeout <= 0:
        raise ValueError("Notebook cell and job timeouts must be positive")
    notebooks = Path(notebooks_dirpath).resolve()
    selected = sorted(set(RESOLUTIONS if resolutions is None else resolutions))
    if not selected or any(r not in RESOLUTIONS for r in selected):
        raise ValueError("Select supported PASSAGE resolutions")
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / MANIFEST_NAME
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        previous = {}
    old_entries = {e["destination"]: e for e in previous.get("notebooks", [])}
    inputs = {}
    if source:
        for path in sorted(source.rglob("*.parquet")):
            if _selected(path.relative_to(source), selected):
                inputs[path.relative_to(source).as_posix()] = _fingerprint(path, content=hash_inputs or execute)
    auxiliary = {}
    if config_filepath is None and (notebooks.parent / "config/config.yaml").is_file():
        config_filepath = notebooks.parent / "config/config.yaml"
    for label, paths in [
        ("config", [config_filepath] if config_filepath else []),
        ("style", [style_filepath] if style_filepath else [Path(__file__).with_name("plotting.py")]),
        ("checkpoint", checkpoint_paths or []),
        ("environment", [notebooks.parent / "pyproject.toml", notebooks.parent / "uv.lock"]),
    ]:
        auxiliary[label] = {str(p): _fingerprint(Path(p)) for p in paths if Path(p).is_file()}
    auxiliary["producer_code"] = {
        p.relative_to(Path(__file__).parent.parent).as_posix(): _fingerprint(p)
        for directory in (Path(__file__).parent, Path(__file__).parent.parent / "passage-eval")
        for p in sorted(directory.rglob("*.py"))
    }
    auxiliary["results"] = (
        {
            p.relative_to(Path(result_root)).as_posix(): _fingerprint(p)
            for p in sorted(Path(result_root).rglob("*"))
            if p.is_file()
        }
        if result_root
        else {}
    )
    qualification_bundle, qualification_files, qualification_evidence = _stage_qualification_evidence(
        qualification_root, staging
    )
    auxiliary["qualification"] = qualification_files
    inventory = []
    for path in sorted(notebooks.rglob("*.ipynb")):
        relative = path.relative_to(notebooks)
        destination = staging / "notebooks" / relative
        # Leading underscores intentionally preserved for source copies.
        write_staged_bytes(destination, path.read_bytes())
        inventory.append(
            {
                "source": str(path),
                "destination": destination.relative_to(staging).as_posix(),
                "source_sha256": _hash(path),
                "role": "source",
            }
        )
    if vision_notebooks_dirpath is None:
        for ancestor in (notebooks.parent, *notebooks.parent.parents):
            candidate = ancestor / "vision/paper/notebooks"
            if candidate.is_dir():
                vision_notebooks_dirpath = candidate
                break
    jobs = [
        (notebooks / f"_{kind}.ipynb", Path(f"{r}x{r}/{kind}.ipynb"), "resolution", r, kind)
        for r in selected
        for kind in TEMPLATES
    ]
    for group, directory in [
        ("paper", paper_notebooks_dirpath or notebooks.parent / "paper/notebooks"),
        ("vision", vision_notebooks_dirpath),
    ]:
        if directory:
            for path in sorted(Path(directory).glob("*.ipynb")):
                jobs.append((path, Path("notebooks") / group / path.name, group, 256, path.stem))
                if group == "vision":
                    dst = staging / "notebooks/sources/vision" / path.name
                    write_staged_bytes(dst, path.read_bytes())
                    inventory.append(
                        {
                            "source": str(path),
                            "destination": dst.relative_to(staging).as_posix(),
                            "source_sha256": _hash(path),
                            "role": "source",
                        }
                    )
    entries = []
    run = executor or execute_notebook
    for template, relative, group, resolution, kind in jobs:
        destination = staging / relative
        artifact_root = staging / "notebook-artifacts" / relative.with_suffix("")
        params = {
            "resolution": resolution,
            "dataset_path": str(source) if source else None,
            "dataset_repo": repo_id,
            "dataset_revision": revision,
            "output_dir": str(artifact_root),
            "result_root": str(Path(result_root).resolve()) if result_root else None,
            "max_samples": 1,
            "classical_budget": 16,
            "max_expansions": 10000,
            "project_path": str(notebooks.parent),
            **dict(parameters or {}),
        }
        params["qualification_root"] = str(qualification_bundle) if qualification_bundle else None
        required_files = [
            str((artifact_root / "_export_receipt.json").relative_to(staging)),
            str((artifact_root / f"{kind}-{resolution}.json").relative_to(staging)),
        ]
        if kind == "dataset_qualification":
            required_files += [
                (artifact_root / name).relative_to(staging).as_posix()
                for name in ("qualification_figures.json", "fig_qualification.pdf", "fig_qualification.png")
            ]
        entry = {
            "source": str(template),
            "destination": relative.as_posix(),
            "group": group,
            "resolution": resolution,
            "sections": _sections(kind, resolution),
            "required_outputs": [relative.as_posix(), *required_files],
            "execution": {"status": "not_run"},
        }
        if not template.is_file():
            for section in entry["sections"].values():
                section.update(status="unsupported", reason=f"Source template missing: {template.name}")
            entry["source_sha256"] = None
            entries.append(entry)
            continue
        entry["source_sha256"] = _hash(template)
        identity = {
            "source_sha256": entry["source_sha256"],
            "inputs": inputs,
            "auxiliary": auxiliary,
            "parameters": params,
            "revision": revision,
            "repo_id": repo_id,
            "metric_definition": "passage-eval canonical costs; stored timings retain original boundary",
            "cell_timeout": cell_timeout,
            "job_timeout": job_timeout,
        }
        entry["identity_sha256"] = _canonical(identity)
        old = old_entries.get(relative.as_posix(), {})
        if execute and not force and old.get("identity_sha256") == entry["identity_sha256"] and destination.is_file():
            try:
                evidence = _execution_evidence(destination)
                outputs = old.get("artifacts", {})
                unchanged = (
                    bool(outputs)
                    and all(p in outputs for p in required_files)
                    and all((staging / p).is_file() and _hash(staging / p) == h for p, h in outputs.items())
                )
                if unchanged and evidence["notebook_sha256"] == old.get("execution", {}).get("notebook_sha256"):
                    entries.append({**old, "reused": True})
                    continue
            except (OSError, ValueError, nbformat.reader.NotJSONError):
                pass
        notebook = _vision_report(template) if group == "vision" else nbformat.read(template, as_version=4)
        # Capture producer statuses without inventing an admission rule from execution alone.
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "import json as _export_json\nfrom pathlib import Path as _ExportPath\n"
                "_ExportPath(output_dir).mkdir(parents=True, exist_ok=True)\n"
                "_ExportPath(output_dir, '_export_receipt.json').write_text(_export_json.dumps("
                "{'report': globals().get('report', {}), 'section_statuses': globals().get('section_statuses', {})}, "
                "default=str, sort_keys=True))",
                metadata={"tags": ["export-receipt"]},
            )
        )
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        if not execute:
            if destination.exists():
                entry["retained_existing_output"] = True
                entry["execution"]["reason"] = (
                    "Execution disabled; previous publication retained pending identity verification"
                )
            else:
                portable = _publication_parameters(params, relative, revision)
                assignments = "\n".join(f"{name} = {value!r}" for name, value in portable.items()) + "\n"
                parameter_cells = [
                    c
                    for c in notebook.cells
                    if c.cell_type == "code" and "parameters" in c.get("metadata", {}).get("tags", [])
                ]
                if parameter_cells:
                    parameter_cells[0].source += "\n" + assignments
                else:
                    notebook.cells.insert(0, nbformat.v4.new_code_cell(assignments, metadata={"tags": ["parameters"]}))
                notebook.metadata["passage"] = {
                    "published_parameters": portable,
                    "execution_parameters": None,
                    "execution_requested": False,
                }
                write_staged_bytes(destination, nbformat.writes(notebook).encode())
            entries.append(entry)
            continue
        artifact_root.mkdir(parents=True, exist_ok=True)
        receipt_path = artifact_root / "_export_receipt.json"
        start = time.monotonic()
        started_ns = time.time_ns()
        with tempfile.TemporaryDirectory(prefix="notebook-execution-", dir=staging) as directory:
            input_path, output_path = Path(directory) / "source.ipynb", Path(directory) / "executed.ipynb"
            nbformat.write(notebook, input_path)
            backup_root = Path(directory) / "reviewed-artifacts"
            shutil.copytree(artifact_root, backup_root, symlinks=True)
            backup_notebook = Path(directory) / "reviewed-notebook.ipynb"
            if destination.is_file():
                shutil.copyfile(destination, backup_notebook)
            try:
                receipt_path.unlink(missing_ok=True)
                run(
                    input_path,
                    output_path,
                    params,
                    cwd=staging,
                    cell_timeout=cell_timeout,
                    job_timeout=job_timeout,
                    offline=source is not None,
                )
                evidence = _execution_evidence(output_path)
                for required in required_files:
                    required_path = staging / required
                    if not required_path.is_file() or required_path.stat().st_mtime_ns < started_ns:
                        raise ValueError(f"Missing fresh required output: {required}")
                collected = json.loads(receipt_path.read_text())
                reported_figures = collected.get("report", {}).get("figures", [])
                source_requires_figures = any(
                    "save_figure(" in c.source for c in notebook.cells if c.cell_type == "code"
                )
                if source_requires_figures and {Path(p).suffix for p in reported_figures} != {".pdf", ".png"}:
                    raise ValueError(f"Expected PNG and PDF figure receipts for {relative}")
                for figure in reported_figures:
                    figure_path = Path(figure)
                    if not figure_path.is_absolute():
                        figure_path = artifact_root / figure_path
                    if artifact_root not in figure_path.resolve().parents or not figure_path.is_file():
                        raise ValueError(f"Missing or external declared figure: {figure}")
                    if figure_path.stat().st_mtime_ns < started_ns:
                        raise ValueError(f"Stale declared figure: {figure}")
                    entry["required_outputs"].append(figure_path.relative_to(staging).as_posix())
                executed_notebook = nbformat.read(output_path, as_version=4)
                published_notebook, parameter_edits = _portable_notebook(executed_notebook, params, relative, revision)
                write_staged_bytes(destination, nbformat.writes(published_notebook).encode())
                entry["execution"] = {
                    **evidence,
                    "executed_notebook_sha256": evidence["notebook_sha256"],
                    "notebook_sha256": _hash(destination),
                    "portable_parameter_edits": parameter_edits,
                    "elapsed_seconds": time.monotonic() - start,
                }
                for cell in entry["execution"]["executed_cells"]:
                    edit = next((e for e in parameter_edits if e["cell_index"] == cell["cell_index"]), None)
                    if edit:
                        cell["published_source_sha256"] = edit["published_source_sha256"]
                        cell["modified_after_execution"] = True
                if receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    entry["producer_receipt"] = receipt
                    for name, status in receipt.get("section_statuses", {}).items():
                        name = {"learned_S2_S4": "learned_s2_s4"}.get(name, name)
                        declared = dict(status) if isinstance(status, dict) else {"status": status}
                        if name == "learned_s2_s4":
                            declared["status"] = receipt.get("report", {}).get("learned_status", declared.get("status"))
                        # Qualification producers use complete for validated receipt success.
                        # Normalize its manifest spelling without changing producer evidence.
                        if declared.get("status") == "complete":
                            declared["status"] = "admitted"
                        declared.setdefault(
                            "reason", "Executed declared producer; scope and prerequisites are recorded in its report"
                        )
                        if declared.get("status") in ("required", "admitted", "not_run", "unsupported"):
                            entry["sections"][name] = {**entry["sections"].get(name, {"required": False}), **declared}
                    if receipt.get("report", {}).get("population") == "bounded_core_metadata":
                        entry["sections"]["bounded_core_metadata"] = {
                            "required": False,
                            "status": "admitted",
                            "reason": "Bounded smoke census only; full population census remains required",
                        }
                        entry["sections"]["core_census"] = {
                            "required": True,
                            "status": "not_run",
                            "reason": "Full four-split core population census was not executed",
                        }
                entry["artifacts"] = {
                    p.relative_to(staging).as_posix(): _hash(p) for p in sorted(artifact_root.rglob("*")) if p.is_file()
                }
            except BaseException as exc:
                # Restore the complete reviewed artifact bundle if any attempt fails.
                shutil.rmtree(artifact_root)
                os.replace(backup_root, artifact_root)
                if backup_notebook.is_file():
                    write_staged_bytes(destination, backup_notebook.read_bytes())
                else:
                    destination.unlink(missing_ok=True)
                entry["execution"] = {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - start,
                }
                entries.append(entry)
                _write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "notebooks": entries,
                        "source_inventory": inventory,
                        "inputs": inputs,
                        "auxiliary": auxiliary,
                        "content_verified": False,
                        "input_verification_scope": "direct_source_fingerprints",
                        "qualification_evidence": qualification_evidence,
                    },
                )
                raise
        entries.append(entry)
    manifest = {
        "schema_version": 1,
        "source_root": str(source) if source else None,
        "source_repo": repo_id,
        "source_revision": revision,
        "content_verified": False,
        "input_verification_scope": "direct_source_fingerprints",
        "input_verification": "Direct source fingerprints remain provisional; qualification_evidence separately "
        "records validated historical authentication receipts",
        "qualification_evidence": qualification_evidence,
        "inputs": inputs,
        "auxiliary": auxiliary,
        "source_inventory": inventory,
        "notebooks": entries,
        "publication_parquets_status": "staged"
        if (staging / "snapshot-staging-manifest.json").is_file()
        else "not_staged",
        "execution_requested": execute,
        "cell_timeout": cell_timeout,
        "job_timeout": job_timeout,
    }
    _write_json(manifest_path, manifest)
    return manifest
