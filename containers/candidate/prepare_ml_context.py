"""Prepare a private, content-checked ML image kit without editing its source venv."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil

BASE = "python:3.12@sha256:17b9724493d916f56565c131f389faff2597f46dec6ce09516f95125bc9f519f"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def excluded_editable(relative: Path) -> bool:
    name = relative.name
    return (name.startswith("__editable__.passage-") and name.endswith(".pth")) or (
        name.startswith("__editable___passage_") and "_finder" in name
        and (name.endswith(".py") or name.endswith(".pyc"))
    )


def snapshot_site_packages(source: Path, target: Path) -> tuple[dict, list[str]]:
    source, target = source.resolve(), target.absolute()
    if not source.is_dir() or target.exists():
        raise ValueError("source must be a directory and target must be new")
    if target.is_relative_to(source):
        raise ValueError("snapshot cannot be placed inside its source")
    files = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("site-packages snapshot rejects symlinks")
    target.mkdir(parents=True)
    manifest, excluded = {}, []
    for path in files:
        relative = path.relative_to(source)
        if excluded_editable(relative):
            excluded.append(relative.as_posix())
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"unsupported snapshot entry: {relative}")
        before = path.stat()
        try:
            os.link(path, destination)
        except OSError as exc:
            if exc.errno != __import__("errno").EXDEV:
                raise
            shutil.copy2(path, destination)
        value = digest(destination)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed while snapshotting: {relative}")
        manifest[relative.as_posix()] = {"sha256": value, "bytes": after.st_size}
    return manifest, excluded


def verify_files(root: Path, manifest: dict) -> None:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest):
        raise ValueError("snapshot file set differs")
    for relative, record in manifest.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("snapshot path escapes root")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"snapshot digest differs: {relative}")


def prepare(runtime_env: Path, wheel: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("use a new context directory")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("supply the exact separately built core wheel")
    site = runtime_env.resolve() / "lib/python3.12/site-packages"
    distributions = sorted(
        ({"name": item.metadata["Name"], "version": item.version} for item in metadata.distributions(path=[str(site)])),
        key=lambda item: item["name"].lower(),
    )
    output.mkdir(parents=True)
    files, excluded = snapshot_site_packages(site, output / "site-packages")
    write_json(output / "snapshot.files.json", files)
    source = Path(__file__).resolve().parent
    for name in ("ML.Containerfile", "verify_ml_snapshot.py", "verify_ml_core_requirements.py", "verify_ml_runtime.py"):
        shutil.copyfile(source / name, output / name)
    (output / "core-wheels").mkdir()
    shutil.copyfile(wheel, output / "core-wheels" / wheel.name)
    (output / ".dockerignore").write_text(".git\n")
    lock = {
        "schema": "passage-ml-runtime-snapshot/v1",
        "base_image": BASE,
        "platform": "linux/arm64",
        "python_minor": "3.12",
        "source_venv_python": (runtime_env / "pyvenv.cfg").read_text(),
        "source_site_packages": str(site),
        "snapshot_files": "snapshot.files.json",
        "snapshot_files_sha256": digest(output / "snapshot.files.json"),
        "snapshot_file_count": len(files),
        "snapshot_bytes": sum(item["bytes"] for item in files.values()),
        "excluded_editable_files": excluded,
        "distributions": distributions,
        "core_wheel": {"path": "core-wheels/" + wheel.name, "sha256": digest(wheel)},
        "bootstrap_pip": "offline ensurepip wheel bundled in the pinned base; version recorded by image probe",
        "candidate_source": "mounted at /candidate during execution; not copied into this image",
        "lock_semantics": "Exact installed-file snapshot plus pinned base/core wheel; not a wheel-resolution pip lock.",
    }
    write_json(output / "runtime.snapshot.lock.json", lock)
    (output / "runtime.observed.requirements.txt").write_text(
        "# Observed metadata only; install from the verified snapshot kit, not this inventory.\n"
        + "\n".join(f"{item['name']}=={item['version']}" for item in distributions) + "\n"
    )
    checksummed = sorted(path for path in output.rglob("*")
                         if path.is_file() and "site-packages" not in path.relative_to(output).parts)
    (output / "context.SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(output).as_posix()}\n" for path in checksummed
    ))
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = prepare(args.runtime_env, args.core_wheel, args.output)
    print(json.dumps({"context": str(args.output), "files": lock["snapshot_file_count"], "bytes": lock["snapshot_bytes"],
                      "snapshot_files_sha256": lock["snapshot_files_sha256"], "core_wheel_sha256": lock["core_wheel"]["sha256"]}))


if __name__ == "__main__":
    main()
