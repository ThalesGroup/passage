"""Passive, content-addressed candidate packages. Validation never imports a planner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SCHEMA = "passage-candidate-package/v1"
FIELDS = {
    "schema",
    "candidate_id",
    "version",
    "family",
    "entrypoint",
    "kwargs",
    "python",
    "backends",
    "seed",
    "dataset",
    "budget",
    "precision",
    "training",
    "license",
    "image",
    "artifacts",
}
SLUG = re.compile(r"[a-z0-9][a-z0-9.-]{0,95}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path: str | Path) -> dict:
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def artifact_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact requires a relative POSIX path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in {".", ".."} for p in relative.split("/")):
        raise ValueError("artifact path escapes package")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact must be a file inside the package")
    if any((root.joinpath(*parts.parts[:n])).is_symlink() for n in range(1, len(parts.parts) + 1)):
        raise ValueError("materialize artifact symlinks before sealing")
    return path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidatePackage:
    path: Path
    document: dict[str, Any]
    digest: str

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def identity(self) -> str:
        return f"{self.document['candidate_id']}@{self.document['version']}"

    def verify(self) -> None:
        if (
            sha256(canonical_bytes(self.document)).hexdigest() != self.digest
            or load_candidate_package(self.path).digest != self.digest
        ):
            raise ValueError("candidate manifest changed after validation")


def load_candidate_package(path: str | Path) -> CandidatePackage:
    path = Path(path).resolve()
    doc = read_json(path)
    if set(doc) != FIELDS or doc.get("schema") != SCHEMA:
        raise ValueError("candidate package fields or schema differ")
    for name in ("candidate_id", "version", "family"):
        if not isinstance(doc[name], str) or not SLUG.fullmatch(doc[name]):
            raise ValueError(f"invalid {name}")
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", doc["entrypoint"]):
        raise ValueError("entrypoint must name module:Class")
    if not isinstance(doc["kwargs"], dict) or doc["python"] not in {"3.11", "3.12", "3.13", "3.14"}:
        raise ValueError("invalid kwargs or supported Python minor")
    if (
        not isinstance(doc["backends"], list)
        or not doc["backends"]
        or len(set(doc["backends"])) != len(doc["backends"])
        or not set(doc["backends"]) <= {"cpu", "cuda", "openvino", "tensorrt"}
    ):
        raise ValueError("invalid backends")
    if doc["seed"] is not None and (type(doc["seed"]) is not int or doc["seed"] < 0):
        raise ValueError("seed must be a nonnegative integer or null")
    dataset = doc["dataset"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"id", "revision", "config", "cost_model", "cost_weight"}
        or not isinstance(dataset["id"], str)
        or not dataset["id"]
        or not re.fullmatch(r"[0-9a-f]{40}", dataset["revision"])
        or not re.fullmatch(r"[1-9]\d*x[1-9]\d*", dataset["config"])
    ):
        raise ValueError("dataset must identify an immutable revision and configuration")
    from passage_eval.protocol.contracts import BudgetSpec, GraphSpec

    GraphSpec(cost_model=dataset["cost_model"], cost_weight=dataset["cost_weight"])
    if not isinstance(doc["budget"], dict) or not set(doc["budget"]) <= {
        "wall_time_ms",
        "max_expansions",
        "max_queue_operations",
        "max_live_states",
    }:
        raise ValueError("unsupported budget fields; put algorithm limits in candidate kwargs")
    BudgetSpec(**doc["budget"])
    for field in ("precision", "license"):
        if not isinstance(doc[field], str) or not doc[field].strip():
            raise ValueError(f"missing {field}")
    if (
        not isinstance(doc["training"], dict)
        or not doc["training"].get("test_usage")
        or doc["training"].get("split") not in {"train", "none"}
    ):
        raise ValueError("training provenance must declare split and previous test use")
    image = doc["image"]
    if not isinstance(image, dict) or set(image) != {"recipe", "reference"}:
        raise ValueError("image needs recipe and optional immutable reference")
    if image["reference"] is not None and not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image["reference"]):
        raise ValueError("image reference must be a manifest digest, or null before build")
    artifacts = doc["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("package has no artifacts")
    names, roles = set(), set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "role"}
            or not isinstance(item["sha256"], str)
            or not HEX.fullmatch(item["sha256"])
            or item["role"] not in {"code", "weights", "config", "lock", "recipe", "notice"}
        ):
            raise ValueError("invalid artifact record")
        if item["path"] in names:
            raise ValueError("duplicate artifact path")
        names.add(item["path"])
        roles.add(item["role"])
        actual = file_sha256(artifact_path(path.parent, item["path"]))
        if actual != item["sha256"]:
            raise ValueError(f"artifact hash differs: {item['path']}")
    if not {"code", "lock", "recipe"} <= roles or image["recipe"] not in names:
        raise ValueError("code, dependency lock and identified image recipe are required")
    # This also rejects nonfinite values nested in kwargs/provenance.
    return CandidatePackage(path, doc, sha256(canonical_bytes(doc)).hexdigest())
