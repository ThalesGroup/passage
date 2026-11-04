"""Synthetic worlds for the campaign and test admission tests (not a test module).

Everything is built in temporary directories from the tiny synthetic Parquet shards of ``test_roster.py``. A split
named ``test`` here is a synthetic fixture, never the released test split: no file of the real dataset is read.

* ``reference_world(root, split)``: a pinned dataset root, a source roster, the reference admission policy,
  inventory, admissible roster and chain, and a validation warm-up roster disjoint from the admissible roster.
* ``green_world()``: a complete ``passage-test-admission/v1`` record whose six gates are GREEN, with its evidence
  directory (host target only), built once per process; ``green_grant()`` is its admission grant. Fixtures whose
  synthetic split is named ``test`` use that grant to pass the guards.

The world builder lifts the test-split guards of the roster builder and of the reference admission only while it
builds its own synthetic ``test`` rosters (``pytest.MonkeyPatch.context``); the tests under check never do.
"""

from __future__ import annotations

import atexit
import copy
import functools
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
BLOCK_SALT = "synthetic-test-v1/r0003/roster-v1/blocks-0/"  # first six rows: 10, 4, 5, 1, 3, 6 (all placed)
WARM_SALT = "synthetic-warmup/r0003/"
FULL_FILES = [range(0, 5), range(5, 9), range(9, 11)]  # rows 0-10: every row has a footprint
WARMUP_IDS = (100, 101, 102, 103)  # validation rows with fresh content (tensor values differ from rows 0-11)


def load_test_module(name: str, filename: str):
    """Load a sibling test file once per process (``--import-mode=importlib`` has no package imports)."""
    key = f"passage_eval_tests_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def roster_fixtures():
    return load_test_module("roster_fixtures", "test_roster.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True))
    return _sha(path)


def ref(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


EXACT = 2 * 30 * math.sqrt(2)  # the analytic optimum of the flat 3x3 crops: the diagonal


def _exact(row: dict, cost: float = EXACT) -> dict:
    """The row with its stored label cost set to ``cost`` (exact by default, so its reference qualifies)."""
    metadata = copy.deepcopy(row["metadata"])
    metadata["paths"] = {"astar_elevation": {"obstacles": {"cost": cost}, "free": {"cost": cost}}}
    return dict(row, metadata=metadata)


def _fresh_row(TR, number: int, split: str) -> dict:
    """A row with the footprint of row ``number % 11`` and content (tensor, id, index) of its own."""
    from passage.utils import encode_tensor_blob

    row = _exact(TR._row(number % 11, split=split))
    row["id"] = f"{number:010d}"
    row["tensor"] = encode_tensor_blob(TR._tensor(number))
    row["metadata"] = dict(row["metadata"], index=number)
    return row


def write_dataset(TR, root: Path, *, split: str, files, disagreeing=()) -> Path:
    """Synthetic shards like ``test_roster.write_dataset``, with exact stored costs (``disagreeing`` rows: 84.85)."""
    shard_dir = root / "3x3" / split
    shard_dir.mkdir(parents=True)
    listed = []
    for number, members in enumerate(files):
        rows = [_exact(TR._row(i, split=split), 84.85 if i in disagreeing else EXACT) for i in members]
        path = shard_dir / f"data-3-{number:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
        data = path.read_bytes()
        listed.append(
            {
                "path": f"3x3/{split}/{path.name}",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "hub_blob_id": "b" * 40,
            }
        )
    TR.write_receipt(root, listed)
    return root


def add_validation_rows(TR, dataset: Path, numbers=WARMUP_IDS) -> None:
    """Add a validation shard of fresh rows to a pinned root and list it in ``PINNED.json``."""
    receipt = json.loads((dataset / "PINNED.json").read_text())
    shard = dataset / "3x3" / "validation" / "data-3-0100.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([_fresh_row(TR, n, "validation") for n in numbers]), shard)
    data = shard.read_bytes()
    receipt["files"].append(
        {
            "path": "3x3/validation/data-3-0100.parquet",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "hub_blob_id": "c" * 40,
        }
    )
    TR.write_receipt(dataset, receipt["files"])


def sealed_category_table() -> dict:
    from passage_eval.protocol.reference_admission import PROPOSED_CATEGORY_TABLE

    table = copy.deepcopy(PROPOSED_CATEGORY_TABLE)
    table["status"] = "sealed"
    table.pop("note", None)
    return table


def sealed_exclusion_caps() -> dict:
    from passage_eval.protocol.reference_admission import PROPOSED_EXCLUSION_CAPS

    return dict(PROPOSED_EXCLUSION_CAPS, status="sealed")


#: The sealed block analysis of the synthetic worlds (the judge's proposed criterion, sealed for unit tests only).
SEALED_BLOCK_ANALYSIS = {
    "rule": "source_window_union",
    "block_degrees": 5,
    "missing_footprint": "halt",
    "min_blocks": 30,
    "max_block_share": 0.10,
    "max_unplaced_fraction": 0.005,
}
#: The sealed confirmatory analysis of the synthetic worlds (values for unit tests only, sealed by no author).
SEALED_CONFIRMATORY_ANALYSIS = {
    "a": "seg",
    "b": "reg",
    "target": "host",
    "seeds": [1234, 2026, 3407],
    "alpha": 0.05,
    "draws": 10000,
    "permutations": 10000,
    "bootstrap_seed": 101,
    "permutation_seed": 102,
    "latency_bootstrap_seed": 103,
}


def _lifted_guards(TR):
    """A context in which the builder, the reference admission and the warm-up builder accept the synthetic ``test``
    split."""
    from passage_eval.protocol import reference_admission as ra
    from passage_eval.protocol import warmup as warmup_module

    context = pytest.MonkeyPatch.context()
    patch = context.__enter__()
    patch.setattr(TR.BUILD, "refuse_test_split", lambda split, **kwargs: split.strip().lower())
    patch.setattr(ra, "refuse_test_roster", lambda roster, **kwargs: roster["dataset"]["split"])
    patch.setattr(warmup_module, "refuse_test_roster", lambda roster, **kwargs: roster["dataset"]["split"])
    return context


def reference_world(
    root: Path,
    *,
    split: str = "validation",
    full_population: bool | None = None,
    category_table: dict | None = None,
    exclusion_caps: dict | None | str = "default",
    unplaced_block: bool = False,
):
    """A pinned synthetic root with a source roster, its admission chain and a disjoint warm-up roster.

    ``category_table`` replaces the sealed category table of the reference admission policy. By default the test
    world applies the sealed exclusion caps and the validation world, which excludes one task of six, applies none;
    ``exclusion_caps`` replaces them (``None``: no caps). ``unplaced_block`` builds the full test population of rows
    0-11, where row 11 has no footprint and is sealed in the unplaced block.
    """
    from functools import partial

    from passage_eval.protocol import reference_admission as ra
    from passage_eval.protocol.warmup import build_warmup_roster

    TR = roster_fixtures()
    full = (split == "test") if full_population is None else full_population
    root.mkdir(parents=True, exist_ok=True)
    # The validation world excludes one task (row 4: a stored cost that disagrees with the reference).
    dataset = write_dataset(
        TR,
        root / "dataset",
        split=split,
        files=TR.FILES if unplaced_block else FULL_FILES if full else TR.FILES,
        disagreeing=() if split == "test" else (4,),
    )
    if split != "validation":
        add_validation_rows(TR, dataset)
    context = _lifted_guards(TR) if split == "test" else None
    try:
        source, info = TR.BUILD.build_roster(
            dataset,
            benchmark_id=TR.BENCHMARK,
            resolution=3,
            n_expected=12 if unplaced_block else 11 if full else 6,
            salt=BLOCK_SALT,
            split=split,
            require_footprints=True,
            unplaced_block=unplaced_block,
        )
        table = sealed_category_table() if category_table is None else copy.deepcopy(category_table)
        if exclusion_caps == "default":
            caps = sealed_exclusion_caps() if split == "test" else None
        else:
            caps = exclusion_caps
        policy = ra.admission_policy(
            source,
            reference_policy="released_labels",
            reference_max_expansions=9,
            category_table=table,
            exclusion_caps=caps,
        )
        inventory = ra.build_reference_inventory(
            source, partial(ra.load_admission_scenes, dataset_path=dataset), policy
        )
        admissible = ra.admissible_roster(source, inventory)
        chain = ra.admission_chain(source, policy, inventory, admissible)

        def pool(n):
            return TR.BUILD.build_roster(
                dataset,
                benchmark_id="synthetic-warmup-pool",
                resolution=3,
                n_expected=n,
                salt=WARM_SALT,
                split="validation",
                require_footprints=False,
            )[0]

        warmup, selection = build_warmup_roster(
            pool, salt=WARM_SALT, evaluated_roster=admissible, benchmark_id="synthetic-warmup", k=2
        )
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    files = {}
    for name, payload in (
        ("source-roster.json", source),
        ("reference-admission-policy.json", policy),
        ("reference-admission-inventory.json", inventory),
        ("admissible-roster.json", admissible),
        ("reference-admission-chain.json", chain),
        ("warmup-roster.json", warmup),
        ("warmup-selection.json", selection),
    ):
        files[name] = root / "evidence" / name
        write_json(files[name], payload)
    return SimpleNamespace(
        root=root,
        dataset=dataset,
        split=split,
        source=source,
        policy=policy,
        inventory=inventory,
        admissible=admissible,
        chain=chain,
        warmup=warmup,
        selection=selection,
        category_table=table,
        files=files,
        TR=TR,
    )


def harness() -> dict:
    from passage_eval import cli

    return {"commit": cli._checkout_commit(), "source_sha256": cli._source_hash()}


def final_candidates() -> dict:
    from passage_eval.protocol.admission import _default_final_candidates

    return json.loads(_default_final_candidates().read_text())


SEG_DECODER = "passage_eval.baselines.final_bounded:SegFinalBounded"
REG_DECODER = "passage_eval.baselines.final_bounded:RegFinalBounded"
PREDICTORS = {
    "seg": "passage_inference.final_field_predictors:FinalSegPredictor",
    "reg": "passage_inference.final_field_predictors:FinalRegPredictor",
}


def sealed_decisions(world, pinned_root_sha: str, commit: str, *, block_analysis: dict | None = None) -> dict:
    from passage_eval.io import pinned
    from passage_eval.metrics import latency

    dataset = world.source["dataset"]
    selected_ids = [task["sample_id"] for task in world.source["tasks"]]
    selected_population = {
        "n_rows": len(selected_ids),
        "ids_sha256": hashlib.sha256("".join(f"{sample_id}\n" for sample_id in selected_ids).encode()).hexdigest(),
    }
    values = {
        "dataset_revision": {key: dataset[key] for key in ("repo", "commit", "config", "resolution")},
        "dataset_root_sha256": pinned_root_sha,
        "evaluation_cell": {key: world.source["cell"][key] for key in ("cost_model", "cost_weight", "task_variant")},
        "roster_selection": {
            "rule": world.source["selection"]["key"],
            "salt": world.source["selection"]["salt"],
            "n": world.source["selection"]["n_selected"],
        },
        "population": selected_population,
        "budget": {
            "wall_time_ms": 120000,
            "max_expansions": 65537,
            "max_queue_operations": None,
            "max_live_states": None,
        },
        "timing_profile": "supervised_service_rewarm_v1",
        "timing_boundary_id": "passage-timing-boundary/v5-supervised-ipc-service-rewarm",
        "deadline_grace_ms": 10.0,
        "hard_timeout_ms": 600000.0,
        "startup_timeout_s": 600.0,
        "max_restarts": 100,
        "worker_rlimits": {"nofile": 1024, "fsize_mb": 1024, "as_mb": None, "cpu_s": None},
        "reference_policy": "released_labels",
        "reference_cohort_id": None,
        "reference_max_expansions": 9,
        "reference_category_table_sha256": pinned.canonical_sha256(world.category_table),
        "reference_exclusion_caps": {
            key: sealed_exclusion_caps()[key]
            for key in ("max_total_fraction", "max_block_fraction", "min_block_exclusions")
        },
        "block_analysis": dict(SEALED_BLOCK_ANALYSIS if block_analysis is None else block_analysis),
        "censoring_rule": {"id": latency.RULE_ID},
        "min_warm_requests": 100,
        "cold_start_policy": "report worker start and the first warm-up request separately",
        "restart_and_failed_warmup_policy": "re-warm after each restart; a failed re-warm ends re-warming for the pass",
        "paired_order": {"granularity": "per_batch", "procedures": ["astar", "reg", "seg"], "schedule_seed": 2026},
        "batching": {
            "batch_size": 4,
            "lease_or_timeout_ms": 3.6e9,
            "fixed_overhead_ms": 30000,
            "per_seed_overhead_ms": 1000,
            "per_request_overhead_ms": 50,
            "persistence_ms": 2000,
            "observed_request_ms": {"astar": 30.0, "reg": 1100.0, "seg": 1600.0},
            "safety_margin": 3.0,
        },
        "killed_batch_policy": "a killed batch is an ambiguous case, inspected, never scored",
        "candidate_start_failure_policy": "a candidate that cannot start is a candidate failure",
        "missing_footprint_policy": "a task without footprint halts block analysis",
        "warmup": {"salt": WARM_SALT, "k": 2, "dataset_root_sha256": pinned_root_sha},
        "infrastructure_relaunch_policy": "one traced relaunch per batch after inspection",
        "r7_policy": {
            "max_ambiguous_attempts": 3,
            "single_task_candidate_failures": 2,
            "relaunch_stop_fraction": 0.05,
            "candidate_attributable": ["killed_by_window", "invalid_session"],
        },
        "harness_commit_authorization": {"commit": commit},
        "targets": ["host"],
        "edge_timing_boundary": "not applicable: host target only",
        "infrastructure_failure_policy": "transport or lease loss invalidates the batch",
        "seeds": [2026],
        "strict_comparison": True,
        "confirmatory_analysis": dict(SEALED_CONFIRMATORY_ANALYSIS),
    }
    return {name: {"value": value, "status": "sealed"} for name, value in values.items()}


def write_bundle(evidence: Path, name: str, target: str, checkpoint_sha256: str) -> dict:
    """A bundle manifest stand-in whose lineage names ``checkpoint_sha256`` (the field gate G6 reads)."""
    path = evidence / "bundles" / f"{name}-{target}.json"
    write_json(path, {"task": name, "target": target, "lineage": {"checkpoint_sha256": checkpoint_sha256}})
    return ref(evidence, path)


def candidates_block(evidence: Path, targets=("host",)) -> dict:
    """The record's candidates: frozen checkpoints and decoders, and one bundle file per target under ``evidence``."""
    from passage_eval.protocol import admission

    frozen = final_candidates()
    installed = admission._installed_final_decoders()
    module_sha = hashlib.sha256(installed.read_bytes()).hexdigest()
    tree = admission.decoder_tree_sha256(SEG_DECODER)
    block = {}
    for name, decoder in (("seg", SEG_DECODER), ("reg", REG_DECODER)):
        checkpoint = frozen["candidates"][name]["checkpoint_256"]["sha256"]
        entries = {}
        for target in targets:
            bundle = write_bundle(evidence, name, target, checkpoint)
            entries[target] = {
                "bundle": bundle,
                "bundle_sha256": bundle["sha256"],
                "engine_sha256": None
                if target == "host"
                else hashlib.sha256(f"{name}-{target}-engine".encode()).hexdigest(),
                "container_digest": None
                if target == "host"
                else "sha256:" + hashlib.sha256(f"{name}-{target}-image".encode()).hexdigest(),
            }
        block[name] = {
            "checkpoint_sha256": checkpoint,
            "predictor": PREDICTORS[name],
            "decoder": decoder,
            "decoder_tree_sha256": tree,
            "final_decoders_sha256": module_sha,
            "targets": entries,
        }
    return block


REHEARSAL_PLAN_SHA256 = hashlib.sha256(b"rehearsal plan of the synthetic world").hexdigest()


def rehearsal_aggregate(record: dict, name: str, *, n_warm: int = 120) -> dict:
    """A ``passage-campaign-aggregate/v1`` of the rehearsal of ``name``, with the fields G2 binds."""
    from passage_eval.metrics import latency
    from passage_eval.protocol import admission

    decisions = record["protocol"]["decisions"]
    seeds = decisions["seeds"]["value"]
    block = {
        "rule": latency.RULE_ID,
        "status": latency.RULE_STATUS,
        "n_warm_requests": n_warm,
        "meets_min_warm_requests": n_warm >= decisions["min_warm_requests"]["value"],
    }
    equivalence = {"equal": True, "max_abs_difference": 0.0}
    population = {"n_source": 6, "n_excluded": 1, "n_admissible": 5, "n_scored": 5}
    return {
        "schema": admission.AGGREGATE_SCHEMA,
        "procedure": name,
        "plan_sha256": REHEARSAL_PLAN_SHA256,
        "campaign_id": "validation-rehearsal",
        "split": "validation",
        "target": "host",
        "seeds": list(seeds),
        "seed": seeds[0] if len(seeds) == 1 else None,
        "harness": dict(record["harness"]),
        "admissible_roster_sha256": hashlib.sha256(b"rehearsal admissible roster").hexdigest(),
        "reference_admission_chain_sha256": hashlib.sha256(b"rehearsal admission chain").hexdigest(),
        "budget": decisions["budget"]["value"],
        "timing_profile": decisions["timing_profile"]["value"],
        "population": population,
        "receipts": [
            {"batch_index": index, "completion_sha256": hashlib.sha256(f"{name}-{index}".encode()).hexdigest()}
            for index in range(3)
        ],
        "canonical_equivalence": dict(equivalence, per_seed={str(seed): True for seed in seeds}),
        "latency": dict(block) if len(seeds) == 1 else None,
        "per_seed": {
            str(seed): {"population": population, "canonical_equivalence": dict(equivalence), "latency": dict(block)}
            for seed in seeds
        },
    }


def write_rehearsal(evidence: Path, record: dict, *, n_warm: int = 120) -> dict:
    from passage_eval.protocol import admission

    procedures = {}
    for name in record["protocol"]["decisions"]["paired_order"]["value"]["procedures"]:
        path = evidence / "rehearsal" / f"aggregate-{name}.json"
        write_json(path, rehearsal_aggregate(record, name, n_warm=n_warm))
        procedures[name] = {"aggregate": ref(evidence, path)}
    decisions = record["protocol"]["decisions"]
    receipt = {
        "schema": admission.REHEARSAL_SCHEMA,
        "split": "validation",
        "target": "host",
        "plan_sha256": REHEARSAL_PLAN_SHA256,
        "harness": dict(record["harness"]),
        "candidates": {
            name: {"checkpoint_sha256": record["candidates"][name]["checkpoint_sha256"]} for name in ("seg", "reg")
        },
        "budget": decisions["budget"]["value"],
        "timing_profile": decisions["timing_profile"]["value"],
        "warmup": {"performed": True, "roster_sha256": "d" * 64, "k": 2},
        "rewarm": {"exercised": True, "n_rewarm_events": 1},
        "resume": {"exercised": True, "n_batches_resumed": 1, "n_unplanned_relaunches": 0},
        "deliberate_kill": {
            "exercised": True,
            "n_deliberate_kills": 1,
            "n_planned_relaunches": 1,
            "killed_attempts_scored": 0,
        },
        "procedures": procedures,
    }
    path = evidence / "rehearsal" / "native-chain-rehearsal.json"
    write_json(path, receipt)
    return ref(evidence, path)


def approve(record: dict, *, at: str = "2026-11-01T00:00:00+00:00") -> dict:
    from passage_eval.protocol.admission import record_core_sha256

    record = copy.deepcopy(record)
    record["approval"] = None
    record["approval"] = {
        "approver": "Synthetic Approver",
        "approved_at": at,
        "approved_record_sha256": record_core_sha256(record),
    }
    return record


def build_green(
    root: Path,
    *,
    category_table: dict | None = None,
    exclusion_caps: dict | None | str = "default",
    unplaced_block: bool = False,
    block_analysis: dict | None = None,
):
    """A GREEN admission record and its evidence directory under ``root`` (world options: see reference_world;
    ``block_analysis`` replaces the sealed block analysis decision)."""
    world = reference_world(
        root / "world",
        split="test",
        category_table=category_table,
        exclusion_caps=exclusion_caps,
        unplaced_block=unplaced_block,
    )
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    for name, path in world.files.items():
        shutil.copy2(path, evidence / name)
    document = evidence / "sealed-protocol.md"
    document.write_text("# Sealed protocol of the synthetic world\n\nAll decisions are fixed for unit tests.\n")
    h = harness()
    record = {
        "schema": "passage-test-admission/v1",
        "status": "sealed",
        "split": "test",
        "record_id": "synthetic-green-world",
        "protocol": {
            "version": "1.0",
            "document": ref(evidence, document),
            "decisions": sealed_decisions(
                world, world.source["dataset_root_identity"]["sha256"], h["commit"], block_analysis=block_analysis
            ),
        },
        "harness": h,
        "candidates": candidates_block(evidence),
        "evidence": {
            "native_chain_rehearsal": None,
            "edge": {"thor": None, "orin": None, "tiger": None},
            "references": {
                "source_roster": ref(evidence, evidence / "source-roster.json"),
                "policy": ref(evidence, evidence / "reference-admission-policy.json"),
                "inventory": ref(evidence, evidence / "reference-admission-inventory.json"),
                "admissible_roster": ref(evidence, evidence / "admissible-roster.json"),
                "chain": ref(evidence, evidence / "reference-admission-chain.json"),
            },
            "warmup": {
                "roster": ref(evidence, evidence / "warmup-roster.json"),
                "selection": ref(evidence, evidence / "warmup-selection.json"),
            },
        },
        "approval": None,
    }
    record["evidence"]["native_chain_rehearsal"] = write_rehearsal(evidence, record)
    record = approve(record)
    record_path = evidence / "admission-record.json"
    write_json(record_path, record)
    return SimpleNamespace(root=root, evidence=evidence, record_path=record_path, record=record, world=world)


@functools.lru_cache(maxsize=1)
def green_world():
    root = Path(tempfile.mkdtemp(prefix="passage-admission-world-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return build_green(root)


@functools.lru_cache(maxsize=1)
def green_grant():
    from passage_eval.protocol.admission import grant_test_admission

    world = green_world()
    return grant_test_admission(world.record_path, world.evidence)


def copy_green(target: Path):
    """A private copy of the GREEN world that a test may change."""
    source = green_world()
    shutil.copytree(source.root, target)
    evidence = target / "evidence"
    record_path = evidence / "admission-record.json"
    return SimpleNamespace(
        root=target, evidence=evidence, record_path=record_path, record=json.loads(record_path.read_text())
    )


def save_record(world, record: dict, *, reapprove: bool = True) -> dict:
    record = approve(record) if reapprove else record
    write_json(world.record_path, record)
    world.record = record
    return record


# ------------------------------------------------------------------------------------------------ test preparation


def write_preparation_approval(world, *, readers=None, hours_ago: float = 1.0, hours_left: float = 24.0, **changes):
    """A ``passage-test-preparation-approval/v1`` of the synthetic world's record file, written beside it."""
    import datetime as dt

    from passage_eval.protocol.admission import preparation_approval_template

    now = dt.datetime.now(dt.timezone.utc)
    approval = preparation_approval_template(_sha(world.record_path), readers)
    approval.update(
        approver="Synthetic Approver",
        approved_at=(now - dt.timedelta(hours=hours_ago)).isoformat(),
        expires_at=(now + dt.timedelta(hours=hours_left)).isoformat(),
        statement="Synthetic preparation approval for unit tests.",
    )
    approval.update(changes)
    path = world.root / "preparation-approval.json"
    write_json(path, approval)
    world.approval_path = path
    return path


def preparation_world(target: Path, *, phase: str = "roster", readers=None):
    """A copy of the GREEN world whose G4/G5 artefacts are still to be produced, with a preparation approval.

    ``phase="roster"``: no reference nor warm-up file, and the ``population`` decision still required (the roster
    builder inventories it). ``phase="warmup"``: the references are sealed; only the warm-up files are missing.
    """
    from passage_eval.protocol.admission import _REFERENCE_KEYS

    world = copy_green(target)
    record = copy.deepcopy(world.record)
    record["evidence"]["warmup"] = {"roster": None, "selection": None}
    if phase == "roster":
        record["evidence"]["references"] = dict.fromkeys(_REFERENCE_KEYS)
        record["protocol"]["decisions"]["population"] = {"value": None, "status": "required"}
        # The synthetic rows stay small, but preparation exercises the production
        # S-scope check: a real preparation may create only the sealed 1024-row
        # selection, not a fixture-sized test roster.
        record["protocol"]["decisions"]["roster_selection"]["value"]["n"] = 1024
    elif phase != "warmup":
        raise ValueError(f"unknown phase {phase!r}")
    save_record(world, record)
    write_preparation_approval(world, readers=readers)
    return world


def preparation_grant(world):
    from passage_eval.protocol.admission import grant_test_preparation

    return grant_test_preparation(world.record_path, world.approval_path, world.evidence)


# ------------------------------------------------------------------------------------------------ campaign plans


def file_entry(path: Path, roster: dict | None = None) -> dict:
    entry = {"path": str(path), "file_sha256": _sha(path)}
    if roster is not None:
        entry["roster_sha256"] = roster["roster_sha256"]
    return entry


def spec_payload(name: str, *, inference: str, seeds=(2026,), budget=None, **overrides) -> dict:
    payload = {
        "method_name": f"campaign-{name}",
        "family": "classical" if name == "astar" else name,
        "architecture": "fixture",
        "inference": inference,
        "inference_format": "request",
        "resolution": 3,
        "split": "validation",
        "cost_model": "elevation",
        "cost_weight": 62.4,
        "task_variant": "obstacles",
        "seeds": list(seeds),
        "n_samples": 5,
        "measure_latency": True,
        "latency_warmup": 2,
        "latency_repeats": 1,
        "budget": budget or {"wall_time_ms": 20000, "max_expansions": 10},
        "reference_policy": "released_labels",
        "reference_max_expansions": 9,
    }
    payload.update(overrides)
    return payload


def campaign_world(root: Path, *, procedures=("astar", "seg", "reg"), batch_size: int = 2, astar=None):
    """A validation campaign: reference world, partition, specs and a pinned plan (``write_plan`` rewrites it).

    ``astar`` may give ``{"inference", "inference_kwargs", "candidate_dir", "planner_allow_sha256",
    "environment_allow_sha256"}`` for a run with a real synthetic planner.
    """
    from passage_eval.campaign import partition as partition_io
    from passage_eval.metrics import latency

    world = reference_world(root / "world", split="validation")
    admissible_path = world.files["admissible-roster.json"]
    partition_dir = root / "partition"
    manifest = partition_io.partition(
        admissible_path, _sha(admissible_path), world.admissible["roster_sha256"], partition_dir, batch_size
    )
    candidate = root / "candidate"
    candidate.mkdir()
    frozen = final_candidates()
    budget = {"wall_time_ms": 20000, "max_expansions": 10, "max_queue_operations": None, "max_live_states": None}
    entries = {}
    for name in procedures:
        if name == "astar":
            settings = astar or {}
            payload = spec_payload(
                name,
                inference=settings.get("inference", "passage_eval.baselines.classical:CanonicalAStarPlanner"),
                budget={"wall_time_ms": 20000, "max_expansions": 10},
                inference_kwargs=settings.get("inference_kwargs", {}),
            )
            spec_path = root / "specs" / f"{name}.json"
            write_json(spec_path, payload)
            entries[name] = {
                "spec": {"path": str(spec_path), "sha256": _sha(spec_path)},
                "mode": "planner",
                "predictor": None,
                "decoder": None,
                "decoder_kwargs": {},
                "precision": None,
                "candidate_paths": [str(settings.get("candidate_dir", candidate))],
                "runtime_binds": [],
                "decoder_allow_sha256": [],
                "planner_allow_sha256": settings.get("planner_allow_sha256", ["e" * 64]),
                "environment_allow_sha256": settings.get("environment_allow_sha256", ["a" * 64]),
                "split_procedure_allow_sha256": [],
                "checkpoint_sha256": None,
                "bundle_sha256": None,
            }
        else:
            payload = spec_payload(
                name, inference=PREDICTORS[name], budget={"wall_time_ms": 20000, "max_expansions": 10}
            )
            spec_path = root / "specs" / f"{name}.json"
            write_json(spec_path, payload)
            entries[name] = {
                "spec": {"path": str(spec_path), "sha256": _sha(spec_path)},
                "mode": "split",
                "predictor": PREDICTORS[name],
                "decoder": SEG_DECODER if name == "seg" else REG_DECODER,
                "decoder_kwargs": {},
                "precision": "float32_field",
                "candidate_paths": [str(candidate)],
                "runtime_binds": [],
                "decoder_allow_sha256": ["b" * 64],
                "planner_allow_sha256": [],
                "environment_allow_sha256": ["a" * 64],
                "split_procedure_allow_sha256": ["c" * 64],
                "checkpoint_sha256": frozen["candidates"][name]["checkpoint_256"]["sha256"],
                "bundle_sha256": "d" * 64,
            }
    h = harness()
    plan = {
        "schema": "passage-campaign-plan/v1",
        "campaign_id": "synthetic-validation-campaign",
        "split": "validation",
        "target": "host",
        "purpose": "diagnostic",
        "protocol_record_sha256": "f" * 64,
        "admission": None,
        "edge_transport_evidence": None,
        "dataset_root": str(world.dataset),
        "output_dir": str(root / "out"),
        "rosters": {
            "admissible": file_entry(admissible_path, world.admissible),
            "source_roster_sha256": world.source["roster_sha256"],
            "reference_admission_chain": {
                **file_entry(world.files["reference-admission-chain.json"]),
                "chain_sha256": world.chain["chain_sha256"],
            },
        },
        "reference": {"policy": "released_labels", "cohort_id": None, "max_expansions": 9},
        "partition": {
            "dir": str(partition_dir),
            "manifest_sha256": _sha(partition_dir / "partition-manifest.json"),
            "batch_size": batch_size,
        },
        "budget": budget,
        "timing": {
            "profile": "supervised_service_rewarm_v1",
            "boundary_id": "passage-timing-boundary/v5-supervised-ipc-service-rewarm",
            "deadline_grace_ms": 10.0,
            "hard_timeout_ms": 60000.0,
            "startup_timeout_s": 120.0,
            "max_restarts": 2,
            "rlimits": {"nofile": 1024, "fsize_mb": 1024, "as_mb": None, "cpu_s": None},
        },
        "warmup": {
            "roster": file_entry(world.files["warmup-roster.json"], world.warmup),
            "salt": WARM_SALT,
            "k": 2,
            "requests_per_warm_pass": 2,
        },
        "order": {"schedule_seed": 2026, "granularity": "per_batch", "procedures": sorted(procedures)},
        "batching": {
            "lease_or_timeout_ms": 3.6e7,
            "fixed_overhead_ms": 30000,
            "per_seed_overhead_ms": 1000,
            "per_request_overhead_ms": 50,
            "persistence_ms": 2000,
            "observed_request_ms": {name: 100.0 for name in procedures},
            "safety_margin": 3.0,
        },
        "procedures": entries,
        "harness": {"python": sys.executable, "worker_python": sys.executable, **h},
        "aggregation": {"censoring_rule_id": latency.RULE_ID, "min_warm_requests": 100},
        "seeds": [2026],
        "infrastructure_relaunch_policy": "one traced relaunch per batch after inspection",
        "r7_policy": {
            "max_ambiguous_attempts": 3,
            "single_task_candidate_failures": 2,
            "relaunch_stop_fraction": 0.05,
            "candidate_attributable": ["killed_by_window", "invalid_session"],
        },
    }
    plan_path = root / "plan.json"
    sha = write_json(plan_path, plan)
    return SimpleNamespace(
        root=root, world=world, plan=plan, plan_path=plan_path, sha=sha, manifest=manifest, partition_dir=partition_dir
    )


def write_plan(campaign, plan: dict) -> str:
    campaign.plan = plan
    campaign.sha = write_json(campaign.plan_path, plan)
    return campaign.sha


def admitted_campaign(root: Path):
    """A test-split plan over the GREEN world's admissible roster, repeating every sealed decision and candidate
    identity of its record."""
    from passage_eval.campaign import partition as partition_io

    green = green_world()
    decisions = {name: entry["value"] for name, entry in green.record["protocol"]["decisions"].items()}
    evidence = green.evidence
    admissible_path = evidence / "admissible-roster.json"
    grant = green_grant()
    partition_dir = root / "partition"
    root.mkdir(parents=True)
    batch_size = decisions["batching"]["batch_size"]
    partition_io.partition(
        admissible_path,
        _sha(admissible_path),
        green.world.admissible["roster_sha256"],
        partition_dir,
        batch_size,
        allow_test=True,
        admission=grant,
    )
    campaign = campaign_world(root / "validation")  # reused for its procedure layout only
    plan = copy.deepcopy(campaign.plan)
    t = decisions
    plan.update(
        split="test",
        purpose="verified-candidate-run",
        protocol_record_sha256=_sha(green.record_path),
        admission={
            "record": str(green.record_path),
            "record_sha256": _sha(green.record_path),
            "evidence_dir": str(evidence),
        },
        dataset_root=str(green.world.dataset),
        output_dir=str(root / "out"),
        seeds=t["seeds"],
        budget=t["budget"],
        reference={"policy": t["reference_policy"], "cohort_id": t["reference_cohort_id"], "max_expansions": 9},
        infrastructure_relaunch_policy=t["infrastructure_relaunch_policy"],
        r7_policy=t["r7_policy"],
    )
    plan["rosters"] = {
        "admissible": file_entry(admissible_path, green.world.admissible),
        "source_roster_sha256": green.world.source["roster_sha256"],
        "reference_admission_chain": {
            **file_entry(evidence / "reference-admission-chain.json"),
            "chain_sha256": green.world.chain["chain_sha256"],
        },
    }
    plan["partition"] = {
        "dir": str(partition_dir),
        "manifest_sha256": _sha(partition_dir / "partition-manifest.json"),
        "batch_size": batch_size,
    }
    plan["timing"] = {
        "profile": t["timing_profile"],
        "boundary_id": t["timing_boundary_id"],
        "deadline_grace_ms": t["deadline_grace_ms"],
        "hard_timeout_ms": t["hard_timeout_ms"],
        "startup_timeout_s": t["startup_timeout_s"],
        "max_restarts": t["max_restarts"],
        "rlimits": t["worker_rlimits"],
    }
    plan["warmup"] = {
        "roster": file_entry(evidence / "warmup-roster.json", green.world.warmup),
        "salt": t["warmup"]["salt"],
        "k": t["warmup"]["k"],
        "requests_per_warm_pass": t["warmup"]["k"],
    }
    plan["order"] = dict(t["paired_order"])
    plan["batching"] = {key: value for key, value in t["batching"].items() if key != "batch_size"}
    plan["aggregation"] = {"censoring_rule_id": t["censoring_rule"]["id"], "min_warm_requests": t["min_warm_requests"]}
    candidates = green.record["candidates"]
    tree = candidates["seg"]["decoder_tree_sha256"]
    for name, procedure in plan["procedures"].items():
        spec_path = Path(procedure["spec"]["path"])
        payload = json.loads(spec_path.read_text())
        payload.update(split="test", budget=t["budget"], seeds=t["seeds"], latency_warmup=t["warmup"]["k"])
        target = root / "specs" / f"{name}.json"
        write_json(target, payload)
        procedure["spec"] = {"path": str(target), "sha256": _sha(target)}
        if name == "astar":
            procedure["planner_allow_sha256"] = [tree]
        else:
            procedure.update(
                predictor=candidates[name]["predictor"],
                decoder=candidates[name]["decoder"],
                decoder_allow_sha256=[candidates[name]["decoder_tree_sha256"]],
                checkpoint_sha256=candidates[name]["checkpoint_sha256"],
                bundle_sha256=candidates[name]["targets"]["host"]["bundle_sha256"],
            )
    plan_path = root / "plan.json"
    sha = write_json(plan_path, plan)
    return SimpleNamespace(root=root, plan=plan, plan_path=plan_path, sha=sha, green=green)
