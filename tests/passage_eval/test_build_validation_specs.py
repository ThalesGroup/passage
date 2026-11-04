"""Synthetic generation checks; no dataset or target access."""

import hashlib
import json
import io
import tarfile
from pathlib import Path

import pytest

from passage_eval.campaign.direct import build_validation_specs as builder
from passage_eval.io import pinned


def _write(path, value):
    raw = value if isinstance(value, bytes) else pinned.canonical_json(value)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_missing_cells_block_all_output(tmp_path):
    inventory = _write(tmp_path / "inventory.json", {"cells": []})
    catalog = _write(tmp_path / "catalog.json", {})
    inputs = _write(
        tmp_path / "inputs.json",
        {
            "schema": "passage-validation-spec-inputs/v1",
            "cells": [],
            "catalog": catalog,
            "results_root": str(tmp_path / "results"),
            "ledger": str(tmp_path / "ledger"),
            "cavops": "/bin/cavops",
        },
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="inventory or assembly input missing"):
        builder.build(inventory["path"], inventory["sha256"], inputs["path"], inputs["sha256"], output)
    assert not output.exists()


def test_18_cells_have_distinct_pinned_specs_and_queue(tmp_path, monkeypatch):
    cells = [
        {"target": t, "seed": s, "procedure": p}
        for t in ("orin", "thor", "tiger")
        for s in (1234, 2026, 3407)
        for p in ("seg", "reg")
    ]
    inventory = _write(tmp_path / "inventory.json", {"cells": cells})
    catalog = _write(tmp_path / "catalog.json", {})
    sources = [
        {
            **cell,
            "stage": None,
            "stage_dispatch": None,
            "stage_result_archive": None,
            "configuration": None,
            "relay_template": None,
            "compiled_manifest": None,
        }
        for cell in cells
    ]
    inputs = _write(
        tmp_path / "inputs.json",
        {
            "schema": "passage-validation-spec-inputs/v1",
            "cells": sources,
            "catalog": catalog,
            "results_root": str(tmp_path / "results"),
            "ledger": str(tmp_path / "ledger"),
            "cavops": "/bin/cavops",
        },
    )
    monkeypatch.setattr(builder, "_cell", lambda _inventory, source, _catalog, _root: {"cell": source})
    from passage_eval.campaign.direct import session_queue
    from types import SimpleNamespace

    prepared = _write(tmp_path / "prepared.json", {"index_sha256": "a" * 64, "warmup_index": [], "count": 128})
    config = {"index_sha256": "a" * 64, "warmup_index": []}
    spec = {"prepared_root": str(tmp_path), "prepared_sha256": prepared["sha256"]}
    adapter = SimpleNamespace(
        load_spec=lambda *_args, **_kwargs: (spec, config),
        inspect_spec=lambda *_args: (spec, config),
        _validation_provenance=lambda *_args: None,
    )
    monkeypatch.setattr(session_queue, "_adapter", lambda *_args: ("synthetic", adapter))
    monkeypatch.setattr(session_queue, "validate_queue", lambda plan: plan)
    output = tmp_path / "output"
    digest = builder.build(inventory["path"], inventory["sha256"], inputs["path"], inputs["sha256"], output)
    plan = pinned.read_pinned_json(output / "queue.json", digest)
    assert len(plan["sessions"]) == 18
    assert len({s["spec"] for s in plan["sessions"]}) == 18
    for cell in plan["sessions"]:
        assert pinned.read_pinned_json(cell["spec"], cell["sha256"])["cell"]["target"] == cell["target"]
    with pytest.raises(FileExistsError):
        builder.build(inventory["path"], inventory["sha256"], inputs["path"], inputs["sha256"], output)


def test_staging_receipt_refuses_hash_substitution(tmp_path, monkeypatch):
    from passage_eval.protocol import replicas

    root = "/target/payload"
    bundle = _write(tmp_path / "bundle.json", {"artifacts": {"model.onnx": hashlib.sha256(b"onnx").hexdigest()}})
    _write(tmp_path / "model.onnx", b"onnx")
    checkpoint = _write(tmp_path / "checkpoint", b"checkpoint")
    decoder = _write(tmp_path / "decoder", b"decoder")
    catalog = _write(tmp_path / "catalog", {})
    pins = {
        "bundle": bundle["sha256"],
        "checkpoint": checkpoint["sha256"],
        "decoder": decoder["sha256"],
        "engine": hashlib.sha256(b"onnx").hexdigest(),
    }
    paths = {
        k: f"{root}/{v}"
        for k, v in {
            "bundle": "bundle.json",
            "checkpoint": "checkpoint",
            "decoder": "decoder",
            "engine": "model.onnx",
        }.items()
    }
    config = _write(
        tmp_path / "config.json",
        {
            "target": "tiger",
            "seed": 1234,
            "procedure": "reg",
            "dataset_split": "validation",
            "artifact_paths": paths,
            "artifact_pins": pins,
            "compiled_manifest": None,
            "compiled_sha256": None,
        },
    )
    files = {paths[k]: pins[k] for k in paths}
    stage = _write(
        tmp_path / "stage.json",
        {
            "schema": "passage-replica256-stage/v1",
            "target": "tiger",
            "target_root": root,
            "files": files,
            "test_payload_access": False,
            "inference_used": False,
            "vendor_gpu_used": False,
            "artifact_qualification": "not_asserted",
        },
    )
    template = _write(
        tmp_path / "template.json", {"target": "tiger", "runner_argv": ["python", "--config-sha256", "old"]}
    )
    inv = {
        "target": "tiger",
        "seed": 1234,
        "procedure": "reg",
        "bundle_manifest": bundle,
        "checkpoint": checkpoint,
        "decoder": decoder,
        "export_onnx": {"sha256": pins["engine"]},
    }

    def archive_and_dispatch(ref, suffix):
        raw = Path(ref["path"]).read_bytes()
        archive = tmp_path / (suffix + ".tar")
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("results/stage.json")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        archive_ref = {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        dispatch = _write(
            tmp_path / (suffix + "-dispatch.json"),
            {
                "status": "complete",
                "target": "tiger",
                "drain": "SCOPE_STOPPED",
                "test_payload_access": False,
                "error": None,
                "job": "job:1234567890abcdef",
                "result_sha256": archive_ref["sha256"],
            },
        )
        return archive_ref, dispatch

    archive, dispatch = archive_and_dispatch(stage, "good")
    source = {
        "target": "tiger",
        "seed": 1234,
        "procedure": "reg",
        "stage": stage,
        "stage_dispatch": dispatch,
        "stage_result_archive": archive,
        "configuration": config,
        "relay_template": template,
        "compiled_manifest": None,
    }
    monkeypatch.setattr(builder, "validate_config", lambda value: value)
    monkeypatch.setattr(replicas, "load_replica_catalog", lambda *_args: {})
    monkeypatch.setattr(replicas, "replica_identity", lambda *_args: {"checkpoint_sha256": checkpoint["sha256"]})
    assert builder._cell(inv, source, catalog, tmp_path / "results")["configuration_sha256"] == config["sha256"]
    invalid_dispatch = json.loads(Path(dispatch["path"]).read_bytes())
    invalid_dispatch["job"] = "job:not-valid"
    wrong_job = {**source, "stage_dispatch": _write(tmp_path / "invalid-job.json", invalid_dispatch)}
    with pytest.raises(ValueError, match="job receipt"):
        builder._cell(inv, wrong_job, catalog, tmp_path / "results")
    bad_stage = json.loads(Path(stage["path"]).read_bytes())
    bad_stage["files"][paths["checkpoint"]] = "0" * 64
    source["stage"] = _write(tmp_path / "bad-stage.json", bad_stage)
    source["stage_result_archive"], source["stage_dispatch"] = archive_and_dispatch(source["stage"], "bad")
    with pytest.raises(ValueError, match="staged file missing or digest differs"):
        builder._cell(inv, source, catalog, tmp_path / "results")


def test_real_cell_builder_handles_all_18_and_rejects_substitution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from passage_eval import cli
    from passage_eval.campaign.direct import session_queue
    from passage_eval.protocol import admission, replicas

    decoder = _write(tmp_path / "decoder", b"decoder")
    checkpoints = {
        (p, s): _write(tmp_path / f"checkpoint-{p}-{s}", f"{p}-{s}".encode())
        for p in ("seg", "reg")
        for s in (1234, 2026, 3407)
    }
    frozen = {
        "reference_decoders": {"module_sha256": decoder["sha256"]},
        "candidates": {p: {"checkpoint_256": {"sha256": checkpoints[p, 2026]["sha256"]}} for p in ("seg", "reg")},
    }
    frozen_sha = "1" * 64
    catalog_data = {
        "schema": "passage-candidate-replicas/v1",
        "status": "sealed",
        "final_candidates_sha256": frozen_sha,
        "final_decoders_sha256": decoder["sha256"],
        "replicas": {
            str(s): {
                "status": "sealed",
                **{f"{p}_checkpoint_sha256": checkpoints[p, s]["sha256"] for p in ("seg", "reg")},
            }
            for s in (1234, 3407)
        },
    }
    catalog = _write(tmp_path / "catalog.json", catalog_data)
    monkeypatch.setattr(admission, "committed_final_candidates", lambda *_args: (frozen, frozen_sha))
    monkeypatch.setattr(cli, "_checkout_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        replicas,
        "load_replica_catalog",
        lambda path, sha: replicas.validate_replica_catalog(
            pinned.read_pinned_json(path, sha), final_candidates=frozen, final_candidates_sha256=frozen_sha
        ),
    )
    script = _write(tmp_path / "rewrite.py", b"rewrite")
    prepared = _write(tmp_path / "prepared.json", {"index_sha256": "a" * 64, "warmup_index": [], "count": 128})
    config_stub = {"index_sha256": "a" * 64, "warmup_index": []}
    adapter = SimpleNamespace(
        load_spec=lambda *_args, **_kwargs: (
            {"prepared_root": str(tmp_path), "prepared_sha256": prepared["sha256"]},
            config_stub,
        ),
        inspect_spec=lambda *_args: ({}, config_stub),
        _validation_provenance=lambda *_args: None,
    )
    monkeypatch.setattr(session_queue, "_adapter", lambda *_args: ("synthetic", adapter))
    monkeypatch.setattr(session_queue, "validate_queue", lambda plan: plan)
    inventory_cells, sources = [], []
    for target in ("orin", "thor", "tiger"):
        for seed in (1234, 2026, 3407):
            for procedure in ("seg", "reg"):
                ident = f"{target}-{procedure}-{seed}"
                local = tmp_path / ident
                local.mkdir()
                model = _write(local / "model.onnx", ident.encode())
                bundle = _write(local / "bundle.json", {"artifacts": {"model.onnx": model["sha256"]}})
                engine = _write(local / "engine", (ident + "-engine").encode())
                target_root = f"/workspace/{ident}"
                paths = {
                    "bundle": target_root + "/bundle.json",
                    "checkpoint": target_root + "/checkpoint.pt",
                    "decoder": target_root + "/decoder.py",
                    "engine": target_root + "/engine",
                }
                compiled = None
                grn = None
                files = {
                    paths["bundle"]: bundle["sha256"],
                    paths["checkpoint"]: checkpoints[procedure, seed]["sha256"],
                    paths["decoder"]: decoder["sha256"],
                    target_root + "/model.onnx": model["sha256"],
                }
                if target == "tiger":
                    paths["engine"] = target_root + "/model.onnx"
                    engine = model
                else:
                    files[paths["engine"]] = engine["sha256"]
                    evidence = {"input_archive_sha256": "2" * 64, "result_archive_sha256": "3" * 64}
                    compiled = _write(
                        local / "compiled.json",
                        {
                            "engine": "engine",
                            "engine_sha256": engine["sha256"],
                            "source_bundle_sha256": bundle["sha256"],
                            "build_evidence": evidence,
                        },
                    )
                    files[target_root + "/compiled.json"] = compiled["sha256"]
                    if procedure == "seg":
                        grn = {
                            "rewrite_script": target_root + "/rewrite.py",
                            "input_archive": target_root + "/input.tar",
                            "result_archive": target_root + "/result.tar",
                            "reproduction_root": "/tmp",
                        }
                        files.update(
                            {
                                grn["rewrite_script"]: script["sha256"],
                                grn["input_archive"]: "2" * 64,
                                grn["result_archive"]: "3" * 64,
                            }
                        )
                config = {
                    "schema": "passage-direct-target-validation-prototype/v1"
                    if seed == 2026
                    else "passage-direct-target-validation-replica/v1",
                    "dataset_split": "validation",
                    "target": target,
                    "procedure": procedure,
                    "seed": seed,
                    "artifact_paths": paths,
                    "artifact_pins": {
                        "bundle": bundle["sha256"],
                        "checkpoint": checkpoints[procedure, seed]["sha256"],
                        "decoder": decoder["sha256"],
                        "engine": engine["sha256"],
                    },
                    "compiled_manifest": target_root + "/compiled.json" if compiled else None,
                    "compiled_sha256": compiled["sha256"] if compiled else None,
                    "exploratory_unadmitted_precision": True,
                    "budget": {},
                    "warmup_index": [{"sample_id": "warm"}],
                    "index_sha256": "a" * 64,
                    "checkpoint_size": 32,
                    "grn_evidence": grn,
                }
                config_ref = _write(local / "config.json", config)
                stage = _write(
                    local / "stage.json",
                    {
                        "schema": "passage-replica256-stage/v1",
                        "target": target,
                        "target_root": target_root,
                        "files": files,
                        "test_payload_access": False,
                        "inference_used": False,
                        "vendor_gpu_used": False,
                        "artifact_qualification": "not_asserted",
                    },
                )
                data = Path(stage["path"]).read_bytes()
                archive = local / "result.tar"
                with tarfile.open(archive, "w") as tar:
                    info = tarfile.TarInfo("results/stage.json")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                archive_ref = {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
                dispatch = _write(
                    local / "dispatch.json",
                    {
                        "status": "complete",
                        "target": target,
                        "drain": "SCOPE_STOPPED",
                        "test_payload_access": False,
                        "error": None,
                        "job": "job:1234567890abcdef",
                        "result_sha256": archive_ref["sha256"],
                    },
                )
                template = _write(
                    local / "template.json", {"target": target, "runner_argv": ["python", "--config-sha256", "old"]}
                )
                inventory_cells.append(
                    {
                        "target": target,
                        "seed": seed,
                        "procedure": procedure,
                        "bundle_manifest": bundle,
                        "checkpoint": checkpoints[procedure, seed],
                        "decoder": decoder,
                        "export_onnx": model,
                        "native": {"sha256": engine["sha256"]},
                        "grn_rewrite": {"script": script},
                    }
                )
                sources.append(
                    {
                        "target": target,
                        "seed": seed,
                        "procedure": procedure,
                        "stage": stage,
                        "stage_dispatch": dispatch,
                        "stage_result_archive": archive_ref,
                        "configuration": config_ref,
                        "relay_template": template,
                        "compiled_manifest": compiled,
                    }
                )
    inventory = _write(tmp_path / "inventory.json", {"cells": inventory_cells})
    inputs_data = {
        "schema": "passage-validation-spec-inputs/v1",
        "cells": sources,
        "catalog": catalog,
        "results_root": str(tmp_path / "results"),
        "ledger": str(tmp_path / "ledger"),
        "cavops": "/bin/cavops",
    }
    inputs = _write(tmp_path / "inputs.json", inputs_data)
    output = tmp_path / "generated"
    digest = builder.build(inventory["path"], inventory["sha256"], inputs["path"], inputs["sha256"], output)
    assert len(pinned.read_pinned_json(output / "queue.json", digest)["sessions"]) == 18
    inventory_cells[0]["checkpoint"] = checkpoints["reg", 1234]
    wrong = _write(tmp_path / "wrong-inventory.json", {"cells": inventory_cells})
    with pytest.raises(ValueError, match="checkpoint differs"):
        builder.build(wrong["path"], wrong["sha256"], inputs["path"], inputs["sha256"], tmp_path / "not-generated")
    assert not (tmp_path / "not-generated").exists()
