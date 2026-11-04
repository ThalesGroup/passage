"""Synthetic dispatch checks, without target access or temporary cleanup."""
import hashlib
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from passage_eval.campaign.direct import session_queue as queue


def _plan(tmp_path):
    cells = [{"target": t, "seed": s, "procedure": p, "spec": str(tmp_path / f"{t}-{s}-{p}.json"),
              "sha256": "a" * 64, "admission": None, "_module": "synthetic", "_output": str(tmp_path / f"out-{t}-{s}-{p}"), "_config_sha256": "b" * 64, "_index_sha256": "c" * 64}
             for t in ("orin", "thor", "tiger") for s in (1234, 2026, 3407) for p in ("seg", "reg")]
    return {"cavops": "/bin/cavops", "sessions": cells}


@pytest.mark.parametrize("lease_busy", [False, True])
def test_one_job_per_cell_and_drained_before_next(tmp_path, lease_busy):
    plan = _plan(tmp_path)
    calls = []
    def run(argv, **_kwargs):
        if argv[1] == "leases":
            return SimpleNamespace(stdout=json.dumps([{"resource": "device:edge-orin"}] if lease_busy else []), returncode=0)
        index = len(calls)
        entry = [e for e in plan["sessions"] if e["target"] == "orin"][index]
        if index:
            assert (tmp_path / f"orin-{index - 1:02d}-complete.json").exists()
        job = f"job:{index + 1:016x}"
        out = Path(entry["_output"])
        out.mkdir()
        scope = f"/scope/{index}"
        drain = {"schema": "passage-direct-session-drain/v1", "target": "orin", "procedure": entry["procedure"],
                 "job_id": job, "scope": scope, "unit_inactive": True, "cgroup_empty": True,
                 "session_processes": [], "handle_evidence": "process_exit_closes_session_descriptors",
                 "runtime_synchronize_close_recorded": True, "driver_context_introspection": False}
        (out / "drain.json").write_text(json.dumps(drain))
        (out / "image-start.json").write_text(json.dumps({"scope": scope, "target": "orin", "procedure": entry["procedure"],
                                                          "configuration_sha256": "b" * 64, "input_index_sha256": "c" * 64}))
        import hashlib
        collected = out / "collected"
        collected.mkdir()
        journal = b"synthetic"
        (collected / "journal.jsonl").write_bytes(journal)
        receipt = json.dumps({"configuration_sha256": "b" * 64, "input_index_sha256": "c" * 64,
                              "journal_sha256": hashlib.sha256(journal).hexdigest()}).encode()
        (collected / "target-receipt.json").write_bytes(receipt)
        (collected / "target-receipt.sha256").write_text(hashlib.sha256(receipt).hexdigest())
        calls.append(argv)
        return SimpleNamespace(stdout=job, returncode=0)
    if lease_busy:
        with pytest.raises(RuntimeError, match="lease not released"):
            queue._run_target(plan, "orin", tmp_path, run=run)
        assert len(calls) == 1
        assert not (tmp_path / "orin-01-started.json").exists()
        return
    queue._run_target(plan, "orin", tmp_path, run=run)
    assert len(calls) == 6
    keys = [a[a.index("--effect-key") + 1] for a in calls]
    assert len(set(keys)) == 6
    assert all(a[a.index("--resource") + 1] == "device:edge-orin" for a in calls)
    assert all(a[a.index("--") + 1:a.index("--") + 5] ==
               ["/usr/bin/timeout", "--signal=TERM", "--kill-after=15s", "1180s"] for a in calls)
    pos = calls[0].index("/usr/bin/timeout")
    subprocess.run(calls[0][pos:pos + 4] + ["/bin/true"], check=True, timeout=2)


def test_failed_or_ambiguous_job_stops_target_without_replay(tmp_path):
    plan = _plan(tmp_path)
    calls = []
    def run(argv, **_kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 1300)
    with pytest.raises(subprocess.TimeoutExpired):
        queue._run_target(plan, "thor", tmp_path, run=run)
    assert len(calls) == 1
    assert (tmp_path / "thor-00-started.json").exists()
    assert (tmp_path / "thor-00-failed.json").exists()
    assert not (tmp_path / "thor-01-started.json").exists()
    with pytest.raises(FileExistsError):
        queue._run_target(plan, "thor", tmp_path, run=run)
    assert len(calls) == 1


def test_three_targets_run_concurrently(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    plan["ledger"] = str(tmp_path / "ledger")
    monkeypatch.setattr(queue, "load_queue", lambda *_args: plan)
    barrier = threading.Barrier(3, timeout=3)
    seen = []
    def run_target(_plan, target, _ledger):
        seen.append(target)
        barrier.wait()
    monkeypatch.setattr(queue, "_run_target", run_target)
    queue.run_queue("unused", "a" * 64)
    assert set(seen) == {"orin", "thor", "tiger"}
    with pytest.raises(FileExistsError):
        queue.run_queue("unused", "a" * 64)


def test_plan_rejects_changed_pairing_before_any_launch(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    admission = {name: "x" for name in queue._ADMISSION}
    admission["warmup_npz_sha256"] = "d" * 64
    admission["scope_sha256"] = "e" * 64
    configs = {}
    specs = {}
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name)
        e["admission"] = dict(admission)
        e["admission"]["record"] = f"{e['target']}-{e['seed']}"
        confpath = e["spec"] + ".config"
        specs[e["spec"]] = {"configuration": confpath, "configuration_sha256": "b" * 64,
                             "output_root": e["spec"] + ".out", "image_manifest_digest": "sha256:" + "f" * 64,
                             "runner_argv": ["python", "--config-sha256", "b" * 64]}
        configs[confpath] = {"dataset_split": "test", "target": e["target"], "seed": e["seed"],
                             "procedure": e["procedure"], "index_sha256": "c" * 64,
                             "warmup_index": [{"sample_id": "warm"}]}
    plan.update(schema="passage-direct-session-queue/v1", ledger=str(tmp_path / "ledger"))
    def read(path, _sha):
        return plan if path == "plan" else specs[path] if path in specs else configs[path]
    def adapter(target):
        if target != "tiger":
            return "synthetic", None
        def inspect(path, _sha):
            spec = specs[path]
            return spec, configs[spec["configuration"]]
        return "synthetic-tiger", SimpleNamespace(inspect_spec=inspect)
    monkeypatch.setattr(queue.pinned, "read_pinned_json", read)
    monkeypatch.setattr(queue, "_adapter", adapter)
    monkeypatch.setattr(queue, "validate_config", lambda config: config)
    from passage_eval.campaign.direct import rule8_scope
    from passage_eval.protocol import admission as admission_module
    monkeypatch.setattr(admission_module, "grant_test_admission", lambda record, _evidence:
                        SimpleNamespace(name=record, record_sha256=hashlib.sha256(record.encode()).hexdigest()))
    class Scope:
        def check_record(self, grant, *, target):
            device, seed = grant.name.split("-")[:2]
            if device != target:
                raise ValueError("record target differs")
            return {"protocol": {"decisions": {"seeds": {"value": [int(seed)]}}}}
    monkeypatch.setattr(rule8_scope, "load_rule8_scope", lambda *_args: Scope())
    checked = queue.load_queue("plan", "a" * 64)
    assert len(checked["sessions"]) == 18
    # Use fresh input because load_queue annotates its parsed plan in memory.
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name)
    configs[plan["sessions"][-1]["spec"] + ".config"]["index_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same input"):
        queue.load_queue("plan", "a" * 64)
    assert not (tmp_path / "ledger").exists()
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name, None)
    configs[plan["sessions"][-1]["spec"] + ".config"]["index_sha256"] = "c" * 64
    specs[plan["sessions"][2]["spec"]]["image_manifest_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="image or runner"):
        queue.load_queue("plan", "a" * 64)
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name, None)
    specs[plan["sessions"][2]["spec"]]["image_manifest_digest"] = "sha256:" + "f" * 64
    plan["sessions"][-1]["admission"]["scope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="global scope differs"):
        queue.load_queue("plan", "a" * 64)
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name, None)
    plan["sessions"][-1]["admission"]["scope_sha256"] = "e" * 64
    plan["sessions"][0]["admission"]["record"] = "thor-1234"
    with pytest.raises(ValueError, match="record target differs"):
        queue.load_queue("plan", "a" * 64)
    plan["sessions"][0]["admission"]["record"] = "orin-3407"
    with pytest.raises(ValueError, match="record seed differs"):
        queue.load_queue("plan", "a" * 64)
    for e in plan["sessions"]:
        for name in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(name, None)
    plan["sessions"][0]["admission"]["record"] = "orin-1234"
    plan["sessions"][1]["admission"]["record"] = "orin-1234-altered"
    with pytest.raises(ValueError, match="same admitted record"):
        queue.load_queue("plan", "a" * 64)


def test_validation_rejects_prepared_index_before_dispatch(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    specs, configs, prepared = {}, {}, {}
    tiger_full = []
    for e in plan["sessions"]:
        for key in ("_module", "_output", "_config_sha256", "_index_sha256"):
            e.pop(key)
        cp, rp = e["spec"] + ".config", e["spec"] + ".prepared"
        specs[e["spec"]] = {"configuration": cp, "configuration_sha256": "b" * 64,
                            "output_root": e["spec"] + ".out", "image_manifest_digest": "sha256:" + "f" * 64,
                            "runner_argv": ["python", "--config-sha256", "b" * 64],
                            "prepared_root": rp, "prepared_sha256": "d" * 64,
                            "replica_catalog_sha256": "e" * 64}
        configs[cp] = {"target": e["target"], "seed": e["seed"], "procedure": e["procedure"],
                       "index_sha256": "c" * 64, "warmup_index": [{"sample_id": "warm"}]}
        prepared[str(Path(rp) / "prepared.json")] = {"count": 128, "index_sha256": "c" * 64,
                                                      "warmup_index": [{"sample_id": "warm"}]}
    plan.update(schema="passage-direct-session-queue/v1", ledger=str(tmp_path / "ledger"))
    def read(path, _sha):
        path = str(path)
        return plan if path == "plan" else prepared[path]
    def adapter(target):
        def load(path, _sha, **_kwargs):
            if target == "tiger":
                tiger_full.append(path)
            spec = specs[path]
            return spec, configs[spec["configuration"]], None, None
        def inspect(path, _sha):
            spec, config, *_ = load(path, _sha)
            return spec, config
        return "synthetic", SimpleNamespace(load_spec=load, inspect_spec=inspect,
                                             _validation_provenance=lambda *_args: None)
    monkeypatch.setattr(queue.pinned, "read_pinned_json", read)
    monkeypatch.setattr(queue, "_adapter", adapter)
    last = plan["sessions"][-1]["spec"]
    prepared[str(Path(specs[last]["prepared_root"]) / "prepared.json")]["index_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prepared validation inputs"):
        queue.load_queue("plan", "a" * 64)
    assert tiger_full  # full image/rootfs/bundle preflight, not only inspect_spec
    assert not (tmp_path / "ledger").exists()
