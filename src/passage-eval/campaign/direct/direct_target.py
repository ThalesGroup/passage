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

"""Direct persistent validation-only target endpoint. Exploratory, never test or qualified."""

from __future__ import annotations
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    if not os.environ.get("NUMBA_CACHE_DIR"):
        raise RuntimeError("NUMBA_CACHE_DIR must be set before starting Python")
import argparse
import hashlib
import json
import time
from passage_eval.io import pinned
from passage_eval.protocol.contracts import BudgetSpec
from passage_eval.campaign.direct.native_preflight import validate_native_files, BACKENDS
from passage_eval.campaign.direct.direct_session import execute_direct_candidate

SCHEMA = "passage-direct-target-validation-prototype/v1"
REPLICA_VALIDATION_SCHEMA = "passage-direct-target-validation-replica/v1"
ADMITTED_SCHEMA = "passage-direct-target-admitted-s/v1"
RULE8_SCHEMA = "passage-direct-target-rule8-s/v1"
KEYS = {
    "schema",
    "dataset_split",
    "target",
    "procedure",
    "seed",
    "artifact_paths",
    "artifact_pins",
    "compiled_manifest",
    "compiled_sha256",
    "exploratory_unadmitted_precision",
    "budget",
    "warmup_index",
    "index_sha256",
    "checkpoint_size",
    "grn_evidence",
}


def validate_config(value):
    if not isinstance(value, dict) or value.get("schema") not in {
        SCHEMA,
        REPLICA_VALIDATION_SCHEMA,
        ADMITTED_SCHEMA,
        RULE8_SCHEMA,
    }:
        raise ValueError("target configuration differs")
    expected_keys = KEYS | {"native_stack"} if value["schema"] == ADMITTED_SCHEMA else KEYS
    if set(value) != expected_keys:
        raise ValueError("target configuration differs")
    if (value["schema"] in {SCHEMA, REPLICA_VALIDATION_SCHEMA} and value["dataset_split"] != "validation") or (
        value["schema"] in {ADMITTED_SCHEMA, RULE8_SCHEMA} and value["dataset_split"] != "test"
    ):
        raise ValueError("configuration split and schema differ; validation requires validation schema")
    if value["schema"] == ADMITTED_SCHEMA and (
        not isinstance(value["native_stack"], dict)
        or set(value["native_stack"]) != {"precision", "runtime"}
        or any(not isinstance(v, str) or not v.strip() for v in value["native_stack"].values())
    ):
        raise ValueError("native precision and runtime facts required")
    allowed_seeds = (
        {2026}
        if value["schema"] == SCHEMA
        else {1234, 3407}
        if value["schema"] == REPLICA_VALIDATION_SCHEMA
        else {1234, 2026, 3407}
    )
    if (
        value["target"] not in BACKENDS
        or value["procedure"] not in ("seg", "reg")
        or value["seed"] not in allowed_seeds
    ):
        raise ValueError("target candidate differs")
    if value["exploratory_unadmitted_precision"] is not (value["schema"] in {SCHEMA, REPLICA_VALIDATION_SCHEMA}):
        raise ValueError("precision mode does not match the session scope")
    if value["procedure"] == "seg" and value["target"] in ("orin", "thor"):
        if not isinstance(value["grn_evidence"], dict) or set(value["grn_evidence"]) != {
            "rewrite_script",
            "input_archive",
            "result_archive",
            "reproduction_root",
        }:
            raise ValueError("Seg GRN evidence required")
    elif value["grn_evidence"] is not None:
        raise ValueError("unexpected GRN evidence")
    if not pinned.is_hex64(value["index_sha256"]) or len(value["warmup_index"]) != 1:
        raise ValueError("input index identity differs")
    BudgetSpec(**value["budget"])
    return value


def execute_stream(stream, config, runtime_factory, decoders, output, *, target_started_ns=None):
    """Consume PSI1, one warm PSF2, then lazy PSF2 blocks and PSE1."""
    from passage_eval.campaign.direct.persistent_transport import read_task_index, read_warmup, blocks_from_stream

    config = validate_config(config)
    budget = BudgetSpec(**config["budget"])
    tasks = read_task_index(stream, config["index_sha256"], max_tasks=1024)
    if config["schema"] in {ADMITTED_SCHEMA, RULE8_SCHEMA} and len(tasks) != 1024:
        raise ValueError("admitted S session requires exactly 1024 tasks")
    warm = read_warmup(stream, budget, config["warmup_index"])
    blocks = blocks_from_stream(stream, tasks, budget)
    output = Path(output)
    if output.exists():
        raise FileExistsError("output already exists")
    output.mkdir(mode=0o700, parents=True)
    result = execute_direct_candidate(
        runtime_factory,
        decoders,
        config["procedure"],
        warm,
        blocks,
        tasks,
        output / "journal.jsonl",
        checkpoint_size=config["checkpoint_size"],
    )
    result["endpoint_wall_ns"] = None if target_started_ns is None else time.perf_counter_ns() - target_started_ns
    return result


def run(config_path, config_sha256, output, stream):
    started = time.perf_counter_ns()
    raw = pinned.read_pinned(Path(config_path), config_sha256)
    config = validate_config(pinned.strict_json_loads(raw, name="direct config"))
    facts = validate_native_files(
        config["procedure"],
        config["target"],
        config["artifact_paths"],
        config["artifact_pins"],
        config["compiled_manifest"],
        config["compiled_sha256"],
        exploratory=config["schema"] == SCHEMA,
        precision_profile=(
            "native_descriptive"
            if config["schema"] == ADMITTED_SCHEMA
            else "native_observed"
            if config["schema"] in {RULE8_SCHEMA, REPLICA_VALIDATION_SCHEMA}
            else "strict"
        ),
        grn_evidence=config["grn_evidence"],
    )
    import numba.core.config as nc

    if not os.environ.get("NUMBA_CACHE_DIR") or nc.CACHE_DIR != os.environ["NUMBA_CACHE_DIR"]:
        raise ValueError("Numba cache was not configured before import")
    from passage_inference.backends import load_runtime
    from passage_inference.final_edge_adapter import _load_decoders

    decoder = _load_decoders(config["artifact_paths"]["decoder"])
    runtime_identity = {}

    def factory():
        compiled = facts["compiled"]
        if compiled is not None:
            engine = Path(config["compiled_manifest"]).parent / compiled["engine"]
            options = {"device_id": 0, "target_identity": compiled["target_identity"]}
        else:
            engine = Path(config["artifact_paths"]["engine"])
            options = {"device": "GPU.0", "compile_options": {"INFERENCE_PRECISION_HINT": "f16"}}
        runtime = load_runtime(BACKENDS[config["target"]], engine, (1, 3, 256, 256), (1, 1, 256, 256), options=options)
        if callable(getattr(runtime, "identity", None)):
            runtime_identity.update(runtime.identity())
        return runtime

    result = execute_stream(stream, config, factory, decoder, output, target_started_ns=started)
    after = validate_native_files(
        config["procedure"],
        config["target"],
        config["artifact_paths"],
        config["artifact_pins"],
        config["compiled_manifest"],
        config["compiled_sha256"],
        exploratory=config["schema"] == SCHEMA,
        precision_profile=(
            "native_descriptive"
            if config["schema"] == ADMITTED_SCHEMA
            else "native_observed"
            if config["schema"] in {RULE8_SCHEMA, REPLICA_VALIDATION_SCHEMA}
            else "strict"
        ),
        grn_evidence=config["grn_evidence"],
    )
    if after["file_hashes"] != facts["file_hashes"]:
        raise ValueError("native artifacts changed")
    journal = Path(output) / "journal.jsonl"
    with journal.open("rb") as f:
        journal_sha = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = {
        "schema": (
            "passage-direct-target-validation-result-prototype/v1"
            if config["schema"] in {SCHEMA, REPLICA_VALIDATION_SCHEMA}
            else "passage-direct-target-rule8-s-result/v1"
            if config["schema"] == RULE8_SCHEMA
            else "passage-direct-target-admitted-s-result/v1"
        ),
        "configuration_sha256": config_sha256,
        "input_index_sha256": config["index_sha256"],
        "journal_sha256": journal_sha,
        "native_artifact_hashes": facts["file_hashes"],
        "runtime_identity": runtime_identity,
        "precision_profile": facts["precision_profile"],
        "precision_note": facts["precision_note"],
        "native_stack": config.get("native_stack"),
        "qualification_status": (
            "exploratory_unadmitted"
            if config["schema"] in {SCHEMA, REPLICA_VALIDATION_SCHEMA}
            else "host_admission_verification_required"
        ),
        "timing": result,
        "scope_gpu_drain_verified": False,
    }
    encoded = pinned.canonical_json(receipt)
    with (Path(output) / "target-receipt.json").open("xb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    with (Path(output) / "target-receipt.sha256").open("x") as f:
        f.write(hashlib.sha256(encoded).hexdigest() + "\n")
        f.flush()
        os.fsync(f.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.config_sha256, args.output, sys.stdin.buffer), sort_keys=True))


if __name__ == "__main__":
    main()
