"""Build a minimal, hashed Docker context from local sources and offline wheels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This copy is shipped with the leaderboard; the source checkouts are siblings
# of that repository, not ancestors of this file.
INFERENCE = HERE.parents[2] / "inference"
PASSAGE = INFERENCE.parent / "passage"
PROFILES = {"thor": "linux/arm64", "orin": "linux/arm64", "tiger": "linux/amd64"}
PINNED = re.compile(r"^[^\s@]+@sha256:[a-f0-9]{64}$")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"missing or symlinked source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"missing or symlinked source: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", "results", "outputs", "tmp"}
               or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is not accepted in build input: {path}")
        # Restrict the context to code/header inputs; do not copy data, scores,
        # credentials, dataset references, notebooks, or generated binaries.
        if path.is_file() and path.suffix in {".py", ".pyi", ".typed", ".c", ".cc", ".cpp", ".h", ".hpp"}:
            copy_file(path, destination / relative)


def prepare(output: Path, wheelhouse: Path, lock: Path, profile: str, base: str,
            base_provenance: Path | None = None) -> dict:
    if profile not in PROFILES or not PINNED.fullmatch(base):
        raise ValueError("select a supported profile and a base image pinned by sha256 digest")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not wheelhouse.is_dir() or not lock.is_file():
        raise ValueError("a wheelhouse and its hashed requirements lock are required")
    # Prevent recursive copying and accidental removal of source inputs on error.
    resolved_output = output.resolve()
    for parent in (PASSAGE.resolve(), INFERENCE.resolve(), wheelhouse.resolve()):
        if resolved_output == parent or parent in resolved_output.parents:
            raise ValueError("build context must be outside source repositories and wheelhouse")
    lock_text = lock.read_text(encoding="utf-8")
    if not lock_text.strip() or "--hash=sha256:" not in lock_text or "onnxruntime" in lock_text.lower():
        raise ValueError("requirements lock must be hashed and must not contain ONNX Runtime")
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels or any("onnxruntime" in p.name.lower() for p in wheels):
        raise ValueError("wheelhouse must contain wheels and no ONNX Runtime wheel")
    output.mkdir(parents=True)
    try:
        for name in ("Dockerfile", "guard_runtime.py", "profiles.json", "entrypoint.py", "final_decoder_smoke.py",
                     "verify_context.py"):
            copy_file(HERE / name, output / name)
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            copy_file(PASSAGE / name, output / "passage" / name)
        copy_tree(PASSAGE / "src", output / "passage" / "src")
        for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "CMakeLists.txt"):
            copy_file(INFERENCE / name, output / "inference" / name)
        for name in ("cpp", "include", "python"):
            copy_tree(INFERENCE / name, output / "inference" / name)
        copy_file(lock, output / "requirements.lock")
        if base_provenance is not None:
            # A locally built base has no vendor digest: ship its provenance record beside the manifest.
            copy_file(base_provenance, output / "base-provenance.json")
        for wheel in wheels:
            copy_file(wheel, output / "wheels" / wheel.name)
        files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        manifest = {"schema": "passage-edge-container-context-v1", "profile": profile,
                    "platform": PROFILES[profile], "base_image": base, "files_sha256": files}
        if base_provenance is not None:
            manifest["base_provenance_sha256"] = files["base-provenance.json"]
        (output / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--base-image", required=True, help="locally available vendor base pinned by @sha256")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path,
                        help="provenance record of a locally built base (copied into the context and hashed in the manifest)")
    parser.add_argument("--tag", help="build locally under this tag after preparing the context")
    args = parser.parse_args()
    manifest = prepare(args.context, args.wheelhouse, args.lock, args.profile, args.base_image, args.base_provenance)
    if args.tag:
        # Docker may pull a missing FROM even with --pull=false. Require the
        # immutable base to be present before starting an offline build.
        subprocess.run(["docker", "image", "inspect", args.base_image], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        subprocess.run(["docker", "build", "--pull=false", "--network=none", "--platform", manifest["platform"],
                        "--build-arg", f"BASE_IMAGE={args.base_image}", "--build-arg", f"TARGET_PROFILE={args.profile}",
                        "--tag", args.tag, str(args.context)], check=True)
    print(json.dumps({"context": str(args.context), "manifest_sha256": digest(args.context / "context-manifest.json"),
                      "profile": args.profile, "built": bool(args.tag)}))


if __name__ == "__main__":
    main()
