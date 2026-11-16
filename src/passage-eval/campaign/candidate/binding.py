"""Bind a completed worker run to an operator-observed OCI execution receipt.

This is a hash-bound evidence chain, not a signature or public admission.
Only the host calls ``bind_container_evidence``, after the worker has exited.
"""

from __future__ import annotations

from pathlib import Path

from passage_eval.candidates.catalog import atomic_json
from passage_eval.protocol.candidate_package import file_sha256, read_json


def _projection(run: dict, receipt: dict, digest: str) -> dict:
    if receipt.get("schema") != "passage-rootless-oci-run/v1":
        raise ValueError("unknown OCI receipt schema")
    if (
        receipt.get("manifest_sha256") != run["manifest_sha256"]
        or receipt.get("input_file_sha256") != run["input_file_sha256"]
        or receipt.get("device") != run["device"]
    ):
        raise ValueError("OCI receipt differs from executed worker identity")
    if (
        receipt.get("returncode") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("rootfs_unchanged") is not True
        or receipt.get("child_namespace_drained") is not True
        or receipt.get("private_runtime_state_empty") is not True
        or receipt.get("cleanup_errors") != []
    ):
        raise ValueError("OCI receipt does not establish completed execution and cleanup")
    return {
        "oci_receipt_sha256": digest,
        "rootfs_receipt_sha256": receipt["rootfs_receipt_sha256"],
        "image_inspect_id": receipt["image_inspect_id"],
        "config_sha256": receipt["config_sha256"],
        "device": receipt["device"],
        "gpu_uuid": receipt.get("gpu_uuid"),
        "strict_admission": receipt["strict_admission"],
    }


def bind_container_evidence(run_dir: str | Path) -> dict:
    """Finalize run.json with the verified host receipt, without modifying it."""
    run_dir = Path(run_dir)
    run = read_json(run_dir / "run.json")
    receipt_path = run_dir / "oci-run.json"
    binding = _projection(run, read_json(receipt_path), file_sha256(receipt_path))
    if "container" in run and run["container"] != binding:
        raise ValueError("run already carries different container evidence")
    run["container"] = binding
    atomic_json(run_dir / "run.json", run)
    return binding


def verify_container_evidence(run_dir: str | Path, run: dict) -> None:
    """Validate an existing host binding before projecting a container result."""
    receipt_path = Path(run_dir) / "oci-run.json"
    expected = _projection(run, read_json(receipt_path), file_sha256(receipt_path))
    if run.get("container") != expected:
        raise ValueError("container receipt binding differs")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    bind_container_evidence(args.run)


if __name__ == "__main__":
    main()
