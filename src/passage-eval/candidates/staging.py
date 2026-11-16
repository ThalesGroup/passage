"""Copy only declared artifacts into a fresh immutable execution snapshot."""

from __future__ import annotations
from pathlib import Path
import shutil

from passage_eval.protocol.candidate_package import CandidatePackage, artifact_path, load_candidate_package


def stage_candidate_package(package: CandidatePackage, destination: str | Path) -> CandidatePackage:
    package.verify()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for record in package.document["artifacts"]:
        source = artifact_path(package.root, record["path"])
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(package.path, destination / "candidate.json")
    staged = load_candidate_package(destination / "candidate.json")
    package.verify()
    if staged.digest != package.digest:
        raise ValueError("staged package identity differs")
    return staged
