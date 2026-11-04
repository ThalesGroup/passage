"""The minimal leased Edge driver is exercised only through a fake CAVIAR subprocess."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.passage_eval import edge_world as world


class FakeCavops:
    """Capture strict argv without ever contacting a device or starting a process."""

    def __init__(self, results: list[subprocess.CompletedProcess[str] | BaseException]):
        self.results = list(results)
        self.calls: list[tuple[list[str], Path, float]] = []

    def __call__(self, argv, *, cwd, timeout, capture_output, text, check):
        assert capture_output is True and text is True and check is False
        self.calls.append((list(argv), Path(cwd), timeout))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _success(job_id: str = "job:0123456789abcdef") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cavops"], 0, stdout=job_id + "\n", stderr="")


def _sealed_root(root: Path, *, target: str, package_sha256: str, role: str):
    from passage_eval.campaign.edge.driver import SealedRoot
    from passage_eval.io import pinned

    root.mkdir()
    raw = pinned.canonical_json(
        {
            "schema": "passage-edge-sealed-root/v1",
            "role": role,
            "target": target,
            "package_sha256": package_sha256,
        }
    )
    (root / "sealed-root.json").write_bytes(raw)
    return SealedRoot(root=root, sha256=pinned.sha256_bytes(raw))


def _invocation(tmp_path: Path, *, target: str = "thor"):
    from passage_eval.campaign.edge.driver import SealedPackageInvocation
    from passage_eval.campaign.edge.package import build_input_package

    package = build_input_package(
        tmp_path / "package",
        target=target,
        plan_sha256=world.PLAN_SHA256,
        procedure="seg",
        inputs=world.inputs(),
    )
    return SealedPackageInvocation(
        package=package,
        adapter_root=_sealed_root(tmp_path / "adapter", target=target, package_sha256=package.sha256, role="adapter"),
        result_root=_sealed_root(tmp_path / "result", target=target, package_sha256=package.sha256, role="result"),
    )


def _idle(attempt, **changes):
    from passage_eval.campaign.edge.driver import IdleCheckEvidence

    values = {
        "target": attempt.target,
        "resource": attempt.resource,
        "job_id": attempt.job_id,
        "effect_key": attempt.effect_key,
        "evidence": {"idle": True, "proof": {"kind": "operator-attestation", "checked_at": "2026-11-03T00:00:00Z"}},
    }
    values.update(changes)
    return IdleCheckEvidence(**values)


def test_leased_driver_uses_the_fixed_sealed_package_entrypoint_and_parses_its_receipt(tmp_path):
    from passage_eval.campaign.edge.driver import CAVIAR_ROOT, EdgeJobDriver

    invocation = _invocation(tmp_path)
    cavops = FakeCavops([_success()])
    driver = EdgeJobDriver(run=cavops)

    receipt = driver.run_package("thor", invocation)

    assert receipt.succeeded is True
    assert receipt.job_id == "job:0123456789abcdef"
    assert receipt.resource == "device:edge-thor"
    assert cavops.calls == [
        (
            [
                "cavops",
                "job-run",
                "--host",
                "edge-thor",
                "--resource",
                "device:edge-thor",
                "--effect-key",
                receipt.effect_key,
                "--effect-fingerprint",
                receipt.effect_fingerprint,
                "--task",
                "passage-edge-qualification",
                "--claim",
                "passage-edge-qualification",
                "--wait",
                "--",
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=30s",
                "1200s",
                "--",
                sys.executable,
                "-I",
                "-B",
                "-m",
                "passage_eval.campaign.edge",
                "sealed-package",
                f"--package={invocation.package.root.resolve()}",
                f"--package-sha256={invocation.package.sha256}",
                f"--adapter-root={invocation.adapter_root.root.resolve()}",
                f"--adapter-root-sha256={invocation.adapter_root.sha256}",
                f"--result-root={invocation.result_root.root.resolve()}",
                f"--result-root-sha256={invocation.result_root.sha256}",
            ],
            CAVIAR_ROOT,
            1230.0,
        )
    ]


def test_sealed_entrypoint_runs_with_active_interpreter_in_isolated_mode(tmp_path):
    from passage_eval.campaign.edge.driver import sealed_package_argv

    argv = sealed_package_argv("thor", _invocation(tmp_path))
    result = subprocess.run(
        [*argv[:6], "--help"], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "sealed-package" in result.stdout


def test_driver_refuses_arbitrary_command_argv_and_tampered_sealed_roots_before_cavops(tmp_path):
    from passage_eval.campaign.edge.driver import EdgeJobDriver

    cavops = FakeCavops([])
    driver = EdgeJobDriver(run=cavops)
    with pytest.raises(ValueError, match="sealed package invocation"):
        driver.run_package("thor", ["/bin/sh", "-c", "payload"])

    invocation = _invocation(tmp_path)
    (invocation.adapter_root.root / "sealed-root.json").write_text("tampered")
    with pytest.raises(ValueError, match="adapter root"):
        driver.run_package("thor", invocation)
    assert cavops.calls == []


def test_leased_cli_rejects_an_arbitrary_remainder_before_a_driver_can_run(monkeypatch):
    from passage_eval.campaign.edge import driver
    from passage_eval.campaign.edge.__main__ import main

    monkeypatch.setattr(driver.EdgeJobDriver, "run_package", lambda *_args: pytest.fail("driver was invoked"))
    with pytest.raises(SystemExit):
        main(["leased-run", "--target", "thor", "--", "/bin/sh", "-c", "payload"])


def test_each_attempt_gets_a_fresh_effect_key(tmp_path):
    from passage_eval.campaign.edge.driver import EdgeJobDriver

    cavops = FakeCavops([_success("job:0000000000000001"), _success("job:0000000000000002")])
    driver = EdgeJobDriver(run=cavops)
    first = driver.run_package("orin", _invocation(tmp_path / "first", target="orin"))
    second = driver.run_package("orin", _invocation(tmp_path / "second", target="orin"))

    assert first.effect_key != second.effect_key
    assert all(key.startswith("passage-edge-orin-") for key in (first.effect_key, second.effect_key))
    assert cavops.calls[0][0][cavops.calls[0][0].index("--effect-key") + 1] == first.effect_key
    assert cavops.calls[1][0][cavops.calls[1][0].index("--effect-key") + 1] == second.effect_key


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("unexpected runner error")])
def test_every_runner_exception_retains_an_ambiguous_attempt_with_its_sealed_scope(tmp_path, error):
    from passage_eval.campaign.edge.driver import EdgeJobDriver

    invocation = _invocation(tmp_path)
    cavops = FakeCavops([error])
    attempt = EdgeJobDriver(run=cavops).run_package("thor", invocation)

    assert attempt.succeeded is False and attempt.lease_retained is True and attempt.returncode == -1
    assert attempt.job_id is None and attempt.invocation == invocation
    assert type(error).__name__ in attempt.stderr
    assert len(cavops.calls) == 1


def test_failed_attempt_is_retained_until_registered_scope_and_full_idle_evidence_allow_release(tmp_path):
    from passage_eval.campaign.edge.driver import EdgeJobDriver, LeaseReleaseError

    cavops = FakeCavops(
        [
            subprocess.CompletedProcess(["cavops"], 17, stdout="job:0123456789abcdef\n", stderr="transport lost"),
            subprocess.CompletedProcess(["cavops"], 0, stdout="released device:edge-tiger\n", stderr=""),
        ]
    )
    driver = EdgeJobDriver(run=cavops)
    attempt = driver.run_package("tiger", _invocation(tmp_path, target="tiger"))

    assert attempt.succeeded is False and attempt.lease_retained is True
    assert len(cavops.calls) == 1, "a failed attempt must not auto-release"
    with pytest.raises(LeaseReleaseError, match="idle-check evidence"):
        driver.release_after_idle(attempt, None)
    with pytest.raises(LeaseReleaseError, match="does not prove idle"):
        driver.release_after_idle(attempt, _idle(attempt, evidence={"idle": True, "proof": {}}))
    with pytest.raises(LeaseReleaseError, match="belongs to another attempt"):
        driver.release_after_idle(attempt, _idle(attempt, resource="device:edge-thor"))

    released = driver.release_after_idle(attempt, _idle(attempt))
    assert released.resource == "device:edge-tiger"
    assert released.idle_evidence["proof"]["kind"] == "operator-attestation"
    assert cavops.calls[1][0] == ["cavops", "release", "device:edge-tiger", "--force"]


def test_release_rejects_forged_or_resource_mismatched_attempts_even_with_idle_evidence(tmp_path):
    from passage_eval.campaign.edge.driver import EdgeJobDriver, LeaseReleaseError

    cavops = FakeCavops([subprocess.CompletedProcess(["cavops"], 7, stdout="job:0123456789abcdef\n", stderr="lost")])
    driver = EdgeJobDriver(run=cavops)
    attempt = driver.run_package("thor", _invocation(tmp_path))

    with pytest.raises(LeaseReleaseError, match="issued by this driver"):
        driver.release_after_idle(replace(attempt), _idle(attempt))
    object.__setattr__(attempt, "resource", "device:edge-forged")
    with pytest.raises(LeaseReleaseError, match="resource differs"):
        driver.release_after_idle(attempt, _idle(attempt))
    assert len(cavops.calls) == 1


def test_driver_uses_configured_lease_command_and_working_directory(tmp_path):
    from passage_eval.campaign.edge.driver import EdgeJobDriver
    coordinator = FakeCavops([_success()])
    EdgeJobDriver(run=coordinator, caviar_root=tmp_path, lease_command='/opt/lease/bin/client').run_package(
        'thor', _invocation(tmp_path)
    )
    assert coordinator.calls[0][0][0] == '/opt/lease/bin/client'
    assert coordinator.calls[0][1] == tmp_path
