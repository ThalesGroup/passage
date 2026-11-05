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

"""Assemble validation relay specs from pinned inventory, configs and real staging receipts.

No target connection or test roster access. All eighteen cells are checked before
writing; a missing or mismatched artifact prevents the entire plan from being made.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

from passage_eval.campaign.direct.direct_target import validate_config
from passage_eval.campaign.direct.oci_validation_relay import _bundle_artifact_paths, _file_digest
from passage_eval.campaign.edge.lease import parse_job_id
from passage_eval.io import pinned


def _ref(value, name):
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value["path"], str)
        or not value["path"]
        or not pinned.is_hex64(value["sha256"])
    ):
        raise ValueError(f"{name}: pinned file reference required")
    if _file_digest(value["path"]) != value["sha256"]:
        raise ValueError(f"{name}: file digest differs")
    return value


def _json_ref(value, name):
    ref = _ref(value, name)
    return pinned.read_pinned_json(ref["path"], ref["sha256"])


def _cell(inventory, source, catalog, results_root):
    from passage_eval.protocol.replicas import load_replica_catalog, replica_identity

    target, seed, procedure = (source[k] for k in ("target", "seed", "procedure"))
    if (target, seed, procedure) != (inventory["target"], inventory["seed"], inventory["procedure"]):
        raise ValueError("inventory and source cell differ")
    stage = _json_ref(source["stage"], "staging receipt")
    dispatch = _json_ref(source["stage_dispatch"], "staging dispatch")
    result_ref = _ref(source["stage_result_archive"], "staging result archive")
    if (
        dispatch.get("status") != "complete"
        or dispatch.get("target") != target
        or dispatch.get("drain") != "SCOPE_STOPPED"
        or dispatch.get("test_payload_access") is not False
        or dispatch.get("error") is not None
        or dispatch.get("result_sha256") != result_ref["sha256"]
    ):
        raise ValueError("staging dispatch is not complete and drained")
    if not isinstance(dispatch.get("job"), str):
        raise ValueError("staging job identity missing")
    parse_job_id(dispatch["job"])
    with tarfile.open(result_ref["path"], "r:*") as archive:
        matches = [m for m in archive.getmembers() if m.name == "results/stage.json" and m.isfile()]
        if (
            len(matches) != 1
            or hashlib.sha256(archive.extractfile(matches[0]).read()).hexdigest() != source["stage"]["sha256"]
        ):
            raise ValueError("stage receipt differs from the drained result archive")
    if (
        stage.get("schema") != "passage-replica256-stage/v1"
        or stage.get("target") != target
        or stage.get("test_payload_access") is not False
        or stage.get("inference_used") is not False
        or stage.get("vendor_gpu_used") is not False
        or stage.get("artifact_qualification") != "not_asserted"
        or not isinstance(stage.get("files"), dict)
    ):
        raise ValueError("staging receipt does not attest the requested validation target")
    config = validate_config(_json_ref(source["configuration"], "target configuration"))
    if (
        config["dataset_split"] != "validation"
        or config["target"] != target
        or config["seed"] != seed
        or config["procedure"] != procedure
    ):
        raise ValueError("target configuration is not this validation cell")
    paths, pins = config["artifact_paths"], config["artifact_pins"]
    if not isinstance(paths, dict) or set(paths) != {"bundle", "checkpoint", "engine", "decoder"}:
        raise ValueError("candidate paths differ")
    stage_root = PurePosixPath(stage["target_root"])
    if not stage_root.is_absolute() or ".." in stage_root.parts:
        raise ValueError("staging target root differs")
    files = stage["files"]

    def staged(path, digest):
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or stage_root not in candidate.parents
            or ".." in candidate.parts
            or files.get(str(candidate)) != digest
        ):
            raise ValueError(f"staged file missing or digest differs: {path}")
        return str(candidate)

    for name, invname in (("bundle", "bundle_manifest"), ("checkpoint", "checkpoint"), ("decoder", "decoder")):
        host = _ref(inventory[invname], invname)
        if pins.get(name) != host["sha256"]:
            raise ValueError(f"{name} differs from the inventory")
        staged(paths[name], host["sha256"])
    replica_catalog = load_replica_catalog(catalog["path"], catalog["sha256"])
    if seed == 2026:
        from passage_eval import cli
        from passage_eval.protocol.admission import committed_final_candidates

        frozen, _ = committed_final_candidates(cli._checkout_commit())
        checkpoint_sha = frozen["candidates"][procedure]["checkpoint_256"]["sha256"]
    else:
        checkpoint_sha = replica_identity(replica_catalog, procedure, seed)["checkpoint_sha256"]
    if checkpoint_sha != pins["checkpoint"]:
        raise ValueError("candidate checkpoint differs from the accepted replica catalogue")
    bundle_ref = inventory["bundle_manifest"]
    bundle = _json_ref(bundle_ref, "bundle")
    members = _bundle_artifact_paths(bundle, paths["bundle"])
    for name, digest in bundle["artifacts"].items():
        member_host = Path(bundle_ref["path"]).parent / name
        if _file_digest(member_host) != digest:
            raise ValueError(f"host bundle member differs: {name}")
        staged(str(PurePosixPath(paths["bundle"]).parent / name), digest)
    mounts = {str(p) for p in members} | {paths["bundle"], paths["checkpoint"], paths["decoder"], paths["engine"]}
    compiled = source["compiled_manifest"]
    if target in ("orin", "thor"):
        comp = _json_ref(compiled, "compiled manifest")
        engine_sha = comp.get("engine_sha256")
        native = inventory["native"]
        expected = native.get("sha256") or (native.get("host_file") or {}).get("sha256")
        if (
            engine_sha != expected
            or engine_sha != pins.get("engine")
            or config["compiled_sha256"] != compiled["sha256"]
            or comp.get("source_bundle_sha256") != pins["bundle"]
        ):
            raise ValueError("compiled engine or bundle differs from inventory")
        staged(paths["engine"], engine_sha)
        staged(config["compiled_manifest"], compiled["sha256"])
        engine_name = comp.get("engine")
        if not isinstance(engine_name, str) or PurePosixPath(engine_name).name != engine_name:
            raise ValueError("compiled engine filename differs")
        adjacent = str(PurePosixPath(config["compiled_manifest"]).parent / engine_name)
        staged(adjacent, engine_sha)
        mounts.update((config["compiled_manifest"], adjacent))
        if procedure == "seg":
            grn = config["grn_evidence"]
            evidence = comp["build_evidence"]
            hashes = {
                "rewrite_script": inventory["grn_rewrite"]["script"]["sha256"],
                "input_archive": evidence["input_archive_sha256"],
                "result_archive": evidence["result_archive_sha256"],
            }
            _ref(inventory["grn_rewrite"]["script"], "rewrite script")
            for key, digest in hashes.items():
                staged(grn[key], digest)
                mounts.add(grn[key])
    else:
        if compiled is not None or config["compiled_manifest"] is not None or config["compiled_sha256"] is not None:
            raise ValueError("Tiger does not use a compiled TensorRT manifest")
        if pins.get("engine") != inventory["export_onnx"]["sha256"]:
            raise ValueError("Tiger ONNX identity differs")
        staged(paths["engine"], pins["engine"])
    relay = _json_ref(source["relay_template"], "relay template")
    if relay.get("target") != target:
        raise ValueError("relay template target differs")
    relay = dict(relay)
    relay.update(
        configuration=source["configuration"]["path"],
        configuration_sha256=source["configuration"]["sha256"],
        bundle_manifest_host=bundle_ref["path"],
        compiled_manifest_host=compiled["path"] if compiled is not None else None,
        target_readonly_mounts=sorted(mounts),
        output_root=str(Path(results_root) / f"{target}-{procedure}-{seed}"),
        replica_catalog=catalog["path"] if seed != 2026 else None,
        replica_catalog_sha256=catalog["sha256"] if seed != 2026 else None,
    )
    argv = list(relay["runner_argv"])
    if argv.count("--config-sha256") != 1 or argv.index("--config-sha256") + 1 >= len(argv):
        raise ValueError("runner configuration argument missing")
    argv[argv.index("--config-sha256") + 1] = source["configuration"]["sha256"]
    relay["runner_argv"] = argv
    return relay


def build(inventory_path, inventory_sha256, inputs_path, inputs_sha256, output):
    inventory = pinned.read_pinned_json(inventory_path, inventory_sha256)
    inputs = pinned.read_pinned_json(inputs_path, inputs_sha256)
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"schema", "cells", "catalog", "results_root", "ledger", "cavops"}
        or inputs["schema"] != "passage-validation-spec-inputs/v1"
    ):
        raise ValueError("spec inputs differ")
    catalog = _ref(inputs["catalog"], "replica catalogue")
    expected = {(t, s, p) for t in ("orin", "thor", "tiger") for s in (1234, 2026, 3407) for p in ("seg", "reg")}

    def keyed(rows):
        if not isinstance(rows, list):
            raise ValueError("cell list missing")
        result = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("cell differs")
            key = (row.get("target"), row.get("seed"), row.get("procedure"))
            if key in result or key not in expected:
                raise ValueError("duplicate or invalid cell")
            result[key] = row
        return result

    inventory_cells, sources = keyed(inventory.get("cells")), keyed(inputs["cells"])
    errors, specs = [], {}
    for key in sorted(expected):
        try:
            if key not in inventory_cells or key not in sources:
                raise ValueError("inventory or assembly input missing")
            row = sources[key]
            if set(row) != {
                "target",
                "seed",
                "procedure",
                "stage",
                "stage_dispatch",
                "stage_result_archive",
                "configuration",
                "relay_template",
                "compiled_manifest",
            }:
                raise ValueError("cell assembly inputs differ")
            specs[key] = _cell(inventory_cells[key], row, catalog, inputs["results_root"])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{key[0]}/{key[1]}/{key[2]}: {exc}")
    if errors:
        raise ValueError("spec generation blocked:\n" + "\n".join(errors))
    out = Path(output)
    out.mkdir(mode=0o700)
    cells = []
    for (target, seed, procedure), spec in sorted(specs.items()):
        raw = pinned.canonical_json(spec)
        path = out / f"{target}-{procedure}-{seed}.json"
        with path.open("xb") as file:
            file.write(raw)
        cells.append(
            {
                "target": target,
                "seed": seed,
                "procedure": procedure,
                "spec": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "admission": None,
            }
        )
    plan = {
        "schema": "passage-direct-session-queue/v1",
        "sessions": cells,
        "ledger": inputs["ledger"],
        "cavops": inputs["cavops"],
    }
    from passage_eval.campaign.direct.session_queue import validate_queue

    validate_queue(pinned.strict_json_loads(pinned.canonical_json(plan), name="proposed queue"))
    raw = pinned.canonical_json(plan)
    with (out / "queue.json").open("xb") as file:
        file.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("inventory", "inventory-sha256", "inputs", "inputs-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(build(args.inventory, args.inventory_sha256, args.inputs, args.inputs_sha256, args.output))


if __name__ == "__main__":
    main()
