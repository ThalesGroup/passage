"""Exercise the frozen decoders across the real supervised worker boundary.

All inputs are synthetic; no dataset, labels or model checkpoints are read.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import passage_eval
from passage_eval.baselines.final_decoders import reg_bellman_descent, seg_logit_astar
from passage_eval.cli import BenchSpec
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest
from passage_eval.supervision import sandbox, treehash
from passage_eval.supervision.redaction import request_token
from passage_eval.supervision.supervisor import SplitSpec, SupervisorConfig, build_runner, inspect_environment

PREDICTOR = """\
import numpy as np
class Predictor:
    def predict(self, request):
        rows, cols = request.elevation.shape
        if request.elevation[0, 0] == 0:
            return np.full((rows, cols), 1.0, dtype=np.float32)
        return np.arange(rows * cols, 0, -1, dtype=np.float32).reshape(rows, cols)
"""


@pytest.fixture
def predictor(tmp_path):
    (tmp_path / "pred.py").write_text(PREDICTOR)
    return tmp_path


@contextlib.contextmanager
def runner(predictor, decoder, isolation, *, module="final_operator_adapters", **config_changes):
    spec = BenchSpec("final-test", "classical", "synthetic", inference="pred:Predictor", inference_format="request")
    split = SplitSpec("pred:Predictor", f"passage_eval.baselines.{module}:{decoder}")
    package = Path(passage_eval.__file__).resolve().parent
    pin = treehash.hash_tree(
        package, role="decoder_entry", symlinks="refuse", strict=True, exclude_dirs=treehash.CANDIDATE_EXCLUDES
    ).sha256
    iso = sandbox.IsolationConfig(level=isolation, env_set={"NUMBA_CACHE_DIR": "/tmp/passage-final-worker-cache"})
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=iso,
        candidate_paths=(str(predictor),),
        decoder_allow_sha256=(pin,),
        startup_timeout_s=120.0,
        hard_timeout_ms=120000.0,
        ipc_baseline_pings=1,
        ipc_baseline_warmup=0,
        **config_changes,
    )
    environment = inspect_environment(config, [split.predictor, split.decoder])
    try:
        with build_runner(spec, split, config, environment, seed=2026) as instance:
            yield instance, pin
    finally:
        environment.close()


def request(direct, budget=None):
    elevation = np.zeros((5, 5), np.float64)
    elevation[0, 0] = 1 if direct else 0
    return PlanRequest(
        "synthetic",
        elevation,
        np.zeros((5, 5), bool),
        (0, 0),
        (4, 4),
        "elevation",
        62.4,
        budget=budget or BudgetSpec(wall_time_ms=60000),
    )


def run_one(instance, req):
    return instance.run_request(req, request_token(instance.secret, 0), 0, "warm")


ISOLATIONS = [
    pytest.param("process", id="process"),
    pytest.param(
        "bwrap", id="bwrap", marks=pytest.mark.skipif(not sandbox.available_bwrap(), reason="bwrap unavailable")
    ),
]


@pytest.mark.parametrize("isolation", ISOLATIONS)
@pytest.mark.parametrize(
    "decoder,direct,fallback",
    [("SegFinalDecoder", True, False), ("RegFinalDecoder", True, False), ("RegFinalDecoder", False, True)],
)
def test_operator_decoder_runs_with_pinned_tree_and_redacted_field(predictor, decoder, direct, fallback, isolation):
    with runner(predictor, decoder, isolation) as (instance, pin):
        record, path = run_one(instance, request(direct))
        assert record["status"] == "ok", record
        assert record["fallback"] is fallback
        assert record["counters_source"] == "operator_decoder"
        assert record["wire_policies"] == ["ordinary-request/v1", "decoder-only/v1"]
        assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (4, 4)
        assert instance.facts["decoder_entry_tree"]["sha256"] == pin
        assert Path(passage_eval.__file__).resolve().parent == Path(instance.facts["decoder_entry_tree"]["path"])
        frozen_module = Path(instance.facts["decoder_entry_tree"]["path"]) / "baselines" / "final_decoders.py"
        assert hashlib.sha256(frozen_module.read_bytes()).hexdigest() == (
            "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
        )
        field = np.arange(25, 0, -1, dtype=np.float32).reshape(5, 5) if direct else np.ones((5, 5), dtype=np.float32)
        reference = seg_logit_astar if decoder == "SegFinalDecoder" else reg_bellman_descent
        expected = reference(field, np.zeros((5, 5), bool), (0, 0), (4, 4))
        np.testing.assert_array_equal(path, expected["path"])


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_reg_worker_rejects_a_global_expansion_budget_below_the_certified_bound(predictor, isolation):
    with runner(predictor, "RegFinalDecoder", isolation) as (instance, _):
        record, path = run_one(instance, request(True, BudgetSpec(wall_time_ms=60000, max_expansions=25)))
        assert record["status"] != "ok"
        assert "certified upper bound 26" in record["diagnostics"]["error"]
        assert not len(path)


@pytest.mark.parametrize(
    "module,decoder,direct",
    [
        ("final_operator_adapters", "RegFinalDecoder", False),
        ("final_bounded", "RegFinalBounded", False),
        ("final_bounded", "RegFinalBounded", True),
        ("final_operator_adapters", "SegFinalDecoder", True),
        ("final_bounded", "SegFinalBounded", True),
    ],
)
def test_protocol_budget_n_plus_one_runs_the_frozen_decoders_under_the_service_profile(
    predictor, module, decoder, direct
):
    """The protocol budget (N + 1 expansions; 65 537 at 256 x 256) reaches both frozen decoders unchanged."""
    budget = BudgetSpec(wall_time_ms=120000.0, max_expansions=26)
    with runner(predictor, decoder, "process", module=module, timing_profile="supervised_service_v1") as (instance, _):
        record, path = run_one(instance, request(direct, budget))
    assert record["status"] == "ok", record
    assert record["counters_source"] == "operator_decoder"
    assert record["timing_boundary"] == "passage-timing-boundary/v4-supervised-ipc-service"
    assert record["work"]["expansions"] <= 26
    field = np.arange(25, 0, -1, dtype=np.float32).reshape(5, 5) if direct else np.ones((5, 5), dtype=np.float32)
    reference = seg_logit_astar if decoder.startswith("Seg") else reg_bellman_descent
    np.testing.assert_array_equal(path, reference(field, np.zeros((5, 5), bool), (0, 0), (4, 4))["path"])
