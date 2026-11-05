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

"""A whole validation campaign with real supervised workers: pause, resume, a deliberate R7 kill, aggregation.

The synthetic planner of ``test_supervision_hardening.py`` runs under bwrap as the ``astar`` procedure of a sealed
plan over the synthetic validation world of ``campaign_world.py`` (five admissible tasks, one excluded, batches of
two). Nothing reads the released dataset.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from passage_eval.campaign import runner as RU
from passage_eval.campaign.aggregate import aggregate_campaign
from passage_eval.supervision import sandbox

HERE = Path(__file__).resolve().parent
needs_bwrap = pytest.mark.skipif(not sandbox.available_bwrap(), reason="unprivileged bwrap unavailable")


def _load(name, filename):
    key = f"passage_eval_tests_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


@needs_bwrap
def test_a_validation_campaign_pauses_resumes_relaunches_a_killed_batch_and_aggregates_like_the_scorer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    W = _load("campaign_world", "campaign_world.py")
    H = _load("supervision_hardening", "test_supervision_hardening.py")
    cand = H.write_candidate(tmp_path / "cand", fake_onnxruntime=False)
    campaign = W.campaign_world(
        tmp_path / "campaign",
        procedures=("astar",),
        astar={
            "inference": "hpkg.entry:Planner",
            "inference_kwargs": {"force_mode": "ok"},
            "candidate_dir": cand,
            "planner_allow_sha256": [H.decoder_pin(cand)],
            "environment_allow_sha256": H.environment_pins(cand, ["hpkg.entry:Planner"]),
        },
    )
    output = Path(campaign.plan["output_dir"])
    paused = RU.run_campaign(campaign.plan_path, campaign.sha, stop_after_batches=1)
    assert paused["outcome"] == "paused" and paused["completed_batches"] == [0]
    # resume, with a deliberate kill of the second batch's session after half a second (R7 exercised on purpose)
    status = RU.run_campaign(campaign.plan_path, campaign.sha, deliberate_kills=[(1, "astar", 0.5)])
    assert status["resumed_batches"] == [0] and status["completed_batches"] == [1, 2]
    assert status["deliberate_kills"] == 1 and status["r7_relaunches"] == 0
    (entry,) = [json.loads(line) for line in (output / "relaunch-log.jsonl").read_text().splitlines()]
    assert (entry["classification"], entry["decision"], entry["unit"]) == ("deliberate_kill", "relaunch", "000001")
    assert entry["drain_proof"]["ok"] is True and entry["drain_proof"]["processes_naming_the_attempt"] == 0
    assert not (output / entry["archived_to"] / "completion.json").exists()

    completion = json.loads((output / "batch-000002" / "completion.json").read_text())
    (window,) = completion["windows"]
    (session,) = window["sessions"]
    assert session["status"] == "complete" and session["scored_receipts"][0]["series"] == "warm"
    assert [r["seed"] for r in session["scored_receipts"]] == [2026]  # one warm series, no cold series
    assert session["cold_start"]["worker_load_ms"]["planner"] > 0
    assert json.loads((output / "batch-000001" / "completion.json").read_text())["attempt"] == 2
    result = aggregate_campaign(campaign.plan_path, campaign.sha, procedure="astar")
    assert result["seeds"] == [2026] and result["seed"] == 2026 and list(result["per_seed"]) == ["2026"]
    assert result["per_seed"]["2026"]["executed_artifact_sha256"] == []  # the synthetic planner loads no checkpoint
    assert result["latency"] == result["per_seed"]["2026"]["latency"]
    assert result["population"] == {"n_source": 6, "n_excluded": 1, "n_admissible": 5, "n_scored": 5}
    assert result["canonical_equivalence"]["equal"] is True
    assert result["split"] == "validation" and result["timing_profile"] == "supervised_service_rewarm_v1"
    assert result["harness"] == {k: campaign.plan["harness"][k] for k in ("commit", "source_sha256")}
    assert result["protocol_record_sha256"] == campaign.plan["protocol_record_sha256"]
    block = result["latency"]
    assert block["n_requests"] == 5 and block["n_distinct_scenes"] == 5 and block["n_repeats"] == 1
    assert block["warmup"]["declared"] is True and block["warmup"]["performed"] is True
    assert block["n_warm_requests"] == 5 and block["meets_min_warm_requests"] is False  # 5 < 100 warm requests
    assert block["meets_min_warm_requests_reason"] == "fewer_warm_requests_than_the_minimum"
    assert result["cold_start"]["n_sessions"] == 3 and result["cold_start"]["first_warmup_request_ms"]["n"] == 3
    assert result["sessions"] == {"n_sessions": 3, "n_closed_by_admission": 0}
    assert result["restarts"]["counted_restarts"] == 0 and result["imputed_failures"]["n"] == 0
    relaunches = result["relaunches"]
    assert relaunches["n_deliberate_kills"] == 1 and relaunches["relaunched_batches"] == []
    assert [row["unit"] for row in result["receipts"]] == ["000000", "000001", "000002"]
    blocks = result["geographic_blocks"]["tasks"]
    assert list(blocks) == [task["sample_id"] for task in campaign.world.admissible["tasks"]]
    assert all(block_id.startswith("geo5:") for block_id in blocks.values())

    # Exercise propagation of checked admission metadata at the aggregation boundary on these
    # synthetic outputs. This stand-in is not a test admission or evidence of a native target run.
    from passage_eval.campaign import plan as plan_module
    from passage_eval.io import pinned

    checked = plan_module.read_plan(campaign.plan_path, campaign.sha)
    grant = SimpleNamespace(report=SimpleNamespace(admitted={"profile": "descriptive_native_2026"}))
    monkeypatch.setattr(plan_module, "read_plan", lambda *args, **kwargs: replace(checked, admission=grant))
    marked = aggregate_campaign(campaign.plan_path, campaign.sha, procedure="astar")
    assert marked["qualification_profile"] == "descriptive_native_2026"
    assert "qualification_profile" not in result
    # The marker is part of the aggregate content covered by the existing input/file digest.
    assert pinned.canonical_sha256(marked) != pinned.canonical_sha256(result)
