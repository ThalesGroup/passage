"""Binding a reviewed split procedure to the actual staged execution inputs."""

from dataclasses import replace
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from passage_eval.cli import BenchSpec
from passage_eval.supervision import PURPOSE_VERIFIED, sandbox
from passage_eval.supervision.supervisor import SplitSpec, SupervisorConfig


def _inputs():
    spec = BenchSpec(
        "test",
        "seg",
        "model",
        inference="ignored:Entry",
        inference_format="request",
        inference_kwargs={"bundle": "/candidate/bundle.json", "backend": "tensorrt"},
    )
    split = SplitSpec("candidate:Predict", "operator:Decode", {"threshold": 1}, "fp16")
    config = SupervisorConfig(
        worker_python="/usr/bin/python3",
        isolation=sandbox.IsolationConfig(level="bwrap"),
        candidate_paths=("/candidate",),
    )
    candidate = SimpleNamespace(path="/candidate", kind="dir", sha256="a" * 64)
    environment = SimpleNamespace(path="/runtime", kind="dir", sha256="b" * 64)
    bound = SimpleNamespace(candidate=[candidate], environment=[environment])
    return spec, split, config, bound


def test_digest_binds_entry_kwargs_artifact_environment_and_precision():
    from passage_eval.supervision.supervisor import _split_procedure_digest

    spec, split, config, bound = _inputs()
    baseline = _split_procedure_digest(spec, split, config, bound)
    assert len(baseline) == 64
    variants = [
        (spec, replace(split, predictor="candidate:Other"), config, bound),
        (
            replace(spec, inference_kwargs={"bundle": "/candidate/other.json", "backend": "tensorrt"}),
            split,
            config,
            bound,
        ),
        (spec, replace(split, decoder_kwargs={"threshold": 2}), config, bound),
        (spec, replace(split, precision="fp32"), config, bound),
        (spec, split, replace(config, candidate_paths=("/other",)), bound),
        (
            spec,
            split,
            config,
            SimpleNamespace(
                candidate=[SimpleNamespace(path="/candidate", kind="dir", sha256="c" * 64)],
                environment=bound.environment,
            ),
        ),
        (
            spec,
            split,
            config,
            SimpleNamespace(
                candidate=bound.candidate, environment=[SimpleNamespace(path="/runtime", kind="dir", sha256="d" * 64)]
            ),
        ),
    ]
    for variant in variants:
        assert _split_procedure_digest(*variant) != baseline
    attested = [{"role": "predictor", "path": "/candidate/model.onnx", "sha256": "e" * 64}]
    assert _split_procedure_digest(spec, split, config, bound, attested_files=attested) != baseline
    assert _split_procedure_digest(spec, split, config, bound, seed=2026) != baseline


def test_procedure_pin_is_optional_unless_explicitly_required():
    from passage_eval.supervision.supervisor import PurposeError, _check_split_procedure_pin

    _, _, config, _ = _inputs()
    digest = "c" * 64
    assert _check_split_procedure_pin(config, digest)["matched"] is False
    assert (
        _check_split_procedure_pin(replace(config, split_procedure_allow_sha256=(digest,)), digest)["matched"] is True
    )
    with pytest.raises(PurposeError, match="split procedure"):
        _check_split_procedure_pin(replace(config, split_procedure_allow_sha256=("d" * 64,)), digest)
    with pytest.raises(PurposeError, match="split procedure"):
        _check_split_procedure_pin(replace(config, purpose=PURPOSE_VERIFIED, require_split_procedure_pin=True), digest)


def test_attest_file_hashes_staged_bytes_and_refuses_unmounted_or_symlink(tmp_path):
    from passage_eval.supervision.supervisor import PurposeError, _attest_split_files

    root = tmp_path / "candidate"
    root.mkdir()
    original = root / "model.onnx"
    original.write_bytes(b"original")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.onnx").write_bytes(b"staged")
    bound = SimpleNamespace(
        candidate=[SimpleNamespace(path=str(root))], environment=[], view=lambda p: staged / p.relative_to(root)
    )
    config = SupervisorConfig(
        worker_python="/usr/bin/python3",
        isolation=sandbox.IsolationConfig(level="bwrap"),
        attest_files=(f"predictor={original}",),
    )
    out = _attest_split_files(config, bound, None)
    assert out[0]["role"] == "predictor" and out[0]["sha256"] == hashlib.sha256(b"staged").hexdigest()
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    with pytest.raises(PurposeError, match="mounted"):
        _attest_split_files(replace(config, attest_files=(f"predictor={outside}",)), bound, None)
    with pytest.raises(PurposeError, match="decoder.*candidate"):
        _attest_split_files(
            replace(config, attest_files=(f"decoder={original}",)),
            bound,
            SimpleNamespace(path=str(tmp_path / "another-decoder")),
        )
    # An overlapping environment mount must not cause us to attest the staged
    # predictor file when the narrowed decoder sees the environment original.
    bound.environment = [SimpleNamespace(path=str(root))]
    with pytest.raises(PurposeError, match="decoder.*candidate"):
        _attest_split_files(
            replace(config, attest_files=(f"decoder={original}",)),
            bound,
            SimpleNamespace(path=str(tmp_path / "another-decoder")),
        )
    bound.environment = []
    link = root / "link"
    link.symlink_to(original)
    with pytest.raises(PurposeError, match="symlink"):
        _attest_split_files(replace(config, attest_files=(f"predictor={link}",)), bound, None)
    fifo = root / "fifo"
    os.mkfifo(fifo)
    staged_fifo = staged / "fifo"
    os.mkfifo(staged_fifo)
    with pytest.raises(PurposeError, match="regular"):
        _attest_split_files(replace(config, attest_files=(f"predictor={fifo}",)), bound, None)


@pytest.mark.skipif(not sandbox.available_bwrap(), reason="bwrap unavailable")
@pytest.mark.parametrize(
    "timing_profile", ["supervised_total_v3", "supervised_service_v1", "supervised_service_rewarm_v1"]
)
def test_operator_pin_matches_actual_split_worker_and_rejects_other_entry(tmp_path, monkeypatch, timing_profile):
    """A reviewed split works under bwrap; another attribute in the same tree is refused."""
    from passage_eval.supervision.supervisor import IntegrityError, PurposeError

    path = Path(__file__).with_name("test_supervision_hardening.py")
    module_spec = importlib.util.spec_from_file_location("supervision_helpers_for_procedure_pin", path)
    helper = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper)
    candidate = helper.write_candidate(tmp_path / "cand")
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    kwargs = {"isolation": "bwrap", "split": split, "kwargs": {}, "timing_profile": timing_profile}
    with helper.runner_for(candidate, **kwargs) as runner:
        digest = runner.facts["split_procedure_pin"]["sha256"]
        assert runner.facts["split_procedure_pin"]["matched"] is False
    required = {"split_procedure_allow_sha256": (digest,), "require_split_procedure_pin": True}
    with helper.runner_for(candidate, **kwargs, **required) as runner:
        assert runner.facts["split_procedure_pin"]["matched"] is True
        assert helper.run_one(runner, helper.request_for())["status"] == "ok"
        monkeypatch.setenv("OMP_NUM_THREADS", "437")
        with pytest.raises(IntegrityError, match="split procedure"):
            runner.identity_check()
    monkeypatch.undo()
    other_split = replace(split, predictor="hpkg.entry:Planner")
    with pytest.raises(PurposeError, match="split procedure"):
        with helper.runner_for(
            candidate, isolation="bwrap", split=other_split, timing_profile=timing_profile, **required
        ):
            pass


def test_service_profiles_require_their_own_reviewed_split_procedure_pin():
    from passage_eval.supervision.supervisor import _split_procedure_digest

    spec, split, config, bound = _inputs()
    legacy = _split_procedure_digest(spec, split, config, bound)
    service = _split_procedure_digest(spec, split, replace(config, timing_profile="supervised_service_v1"), bound)
    rewarm = _split_procedure_digest(spec, split, replace(config, timing_profile="supervised_service_rewarm_v1"), bound)
    assert len({legacy, service, rewarm}) == 3


def test_legacy_split_procedure_digest_is_unchanged_by_the_new_limits():
    """Digests already pinned for the legacy profile stay valid: its descriptor has no limits block."""
    from passage_eval.supervision.supervisor import _split_procedure_digest

    spec, split, config, bound = _inputs()
    legacy = _split_procedure_digest(spec, split, config, bound)
    assert (
        _split_procedure_digest(spec, split, replace(config, startup_timeout_s=60.0, max_restarts=3), bound) == legacy
    )


@pytest.mark.parametrize(
    "config_change,spec_change",
    [
        ({"startup_timeout_s": 60.0}, {}),
        ({"max_restarts": 3}, {}),
        ({"deadline_grace_ms": 20.0}, {}),
        ({"hard_timeout_ms": 300000.0}, {}),
        ({}, {"budget": {"wall_time_ms": 120000.0, "max_expansions": 65537}}),
        ({}, {"budget": {"wall_time_ms": 120000.0, "max_expansions": 65536}}),
    ],
)
def test_service_split_procedure_digest_binds_every_limit(config_change, spec_change):
    from passage_eval.supervision.supervisor import _split_procedure_digest

    spec, split, config, bound = _inputs()
    spec = replace(spec, budget={"wall_time_ms": 120000.0, "max_expansions": 65537})
    config = replace(config, timing_profile="supervised_service_v1")
    baseline = _split_procedure_digest(spec, split, config, bound)
    changed = _split_procedure_digest(replace(spec, **spec_change), split, replace(config, **config_change), bound)
    assert (changed != baseline) is bool(config_change or spec_change["budget"]["max_expansions"] != 65537)


def test_service_split_procedure_pin_keeps_the_v1_limits_and_binds_rlimits_and_frames_itself(monkeypatch):
    """Adding rlimits and frame limits to the procedure identity leaves the reviewed pin digests unchanged."""
    from passage_eval.supervision import supervisor
    from passage_eval.supervision.protocol import FrameLimits
    from passage_eval.protocol.contracts import BudgetSpec

    spec, split, config, bound = _inputs()
    spec = replace(spec, budget={"wall_time_ms": 120000.0, "max_expansions": 65537})
    config = replace(config, timing_profile="supervised_service_v1")
    captured = []
    original = supervisor.json.dumps

    def spy(value, *args, **kwargs):
        if isinstance(value, dict) and str(value.get("schema", "")).startswith("passage-split-procedure-pin/"):
            captured.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(supervisor.json, "dumps", spy)
    baseline = supervisor._split_procedure_digest(spec, split, config, bound)
    monkeypatch.undo()
    (descriptor,) = captured
    assert descriptor["schema"] == "passage-split-procedure-pin/v2"
    assert descriptor["limits"] == supervisor.procedure_limits(
        BudgetSpec(**spec.budget), config, schema=supervisor.PROCEDURE_LIMITS_SCHEMA_V1
    )
    assert "worker_rlimits" not in descriptor["limits"] and "ipc_frame_limits" not in descriptor["limits"]
    # the descriptor binds both through its own entries
    cpu = replace(config, isolation=replace(config.isolation, rlimits=sandbox.RlimitConfig(cpu_seconds=60)))
    frames = replace(config, limits=FrameLimits(max_payload_bytes=1 << 20))
    for changed in (cpu, frames):
        assert supervisor._split_procedure_digest(spec, split, changed, bound) != baseline
