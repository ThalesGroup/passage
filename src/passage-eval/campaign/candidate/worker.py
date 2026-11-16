"""In-container entry point: verify bytes, load a complete planner, and persist raw routes."""

from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

from passage_eval.protocol.candidate_package import load_candidate_package, file_sha256
from passage_eval.candidates.catalog import atomic_json
from passage_eval.campaign.candidate.session import run_candidate_session
from passage_eval.campaign.candidate.transport import load_input_requests


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    package = load_candidate_package(args.manifest)
    if args.device not in package.document["backends"]:
        raise ValueError("backend not declared by candidate")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != package.document["python"]:
        raise ValueError("Python minor differs from sealed candidate")
    # Validate ordinary inputs before importing candidate code.
    requests = list(load_input_requests(args.inputs, package))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (package.root, package.root / "src"):
        sys.path.insert(0, str(path))
    module_name, class_name = package.document["entrypoint"].split(":")
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(package.root):
        raise ValueError("entrypoint resolved outside the sealed candidate package")
    kwargs = dict(package.document["kwargs"])
    kwargs["device"] = args.device
    for name in ("checkpoint", "root"):
        if name in kwargs and isinstance(kwargs[name], str) and not Path(kwargs[name]).is_absolute():
            kwargs[name] = str(package.root / kwargs[name])
    candidate = getattr(module, class_name)(**kwargs)
    summary = None
    try:
        summary = run_candidate_session(
            candidate, requests, manifest=package, journal_path=args.output / "journal.jsonl", device=args.device
        )
    finally:
        sync = getattr(candidate, "synchronize", None)
        try:
            if sync:
                sync()
        finally:
            close = getattr(candidate, "close", None)
            if close:
                close()
    package.verify()
    summary.update(
        schema="passage-candidate-run/v1",
        candidate=package.document,
        manifest_sha256=package.digest,
        input_file_sha256=file_sha256(args.inputs),
        journal_sha256=file_sha256(args.output / "journal.jsonl"),
        synchronization_complete=True,
        runtime_close_complete=True,
        python=sys.version.split()[0],
        device=args.device,
    )
    import resource
    from importlib import metadata

    summary["resources"] = {"max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    torch = sys.modules.get("torch")
    if args.device == "cuda" and torch is not None:
        summary["resources"].update(
            torch_cuda_max_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            torch_cuda_max_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    versions = {}
    for distribution in ("numpy", "scipy", "numba", "torch", "stable-baselines3", "sb3-contrib", "gymnasium"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
    summary["runtime_versions"] = versions
    atomic_json(args.output / "run.json", summary)
    print(json.dumps({"n_completed": summary["n_completed"], "journal_sha256": summary["journal_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
