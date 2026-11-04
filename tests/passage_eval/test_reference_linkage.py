"""The supervisor and the roster scorer use exactly the references of the sealed inventory (judge R1, R6).

The admitted run is the synthetic GREEN world of ``campaign_world.py`` (its split named ``test`` is a synthetic
fixture). The supervisor is stopped right after its reference qualification, before any worker starts.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from passage_eval.protocol.reference_admission import ReferenceIdentityMismatch

HERE = Path(__file__).resolve().parent


def _world():
    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


W = _world()


class Stopped(Exception):
    """Raised in place of the worker environment inspection: the run went past its reference qualification."""


def _supervise(tmp_path, monkeypatch, **spec_changes):
    from passage_eval import cli
    from passage_eval.supervision import sandbox, supervisor

    green = W.green_world()
    admissible = green.evidence / "admissible-roster.json"

    def stop(*args, **kwargs):
        raise Stopped

    monkeypatch.setattr(supervisor, "inspect_environment", stop)
    values = dict(
        method_name="m",
        family="f",
        architecture="a",
        inference="x:Y",
        resolution=3,
        split="test",
        reference_max_expansions=9,
        measure_latency=False,
    )
    values.update(spec_changes)
    roster = supervisor.RosterInput(
        path=str(admissible), sha256=green.world.admissible["roster_sha256"], dataset_root=str(green.world.dataset)
    )
    config = supervisor.SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig())
    return supervisor.run_supervised(
        cli.BenchSpec(**values), tmp_path / "run", config, roster=roster, allow_test=True, admission=W.green_grant()
    )


def test_an_admitted_run_with_the_sealed_limit_passes_the_reference_check(tmp_path, monkeypatch):
    with pytest.raises(Stopped):
        _supervise(tmp_path, monkeypatch)


def test_an_admitted_run_with_another_search_limit_is_refused_before_any_worker(tmp_path, monkeypatch):
    with pytest.raises(ReferenceIdentityMismatch, match="the supervisor: 0 task.*11 reference_id"):
        _supervise(tmp_path, monkeypatch, reference_max_expansions=10)
    assert not (tmp_path / "run").exists()
