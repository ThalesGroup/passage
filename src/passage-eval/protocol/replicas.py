"""Pinned identities of later training replicas, separate from the frozen 2026 definition."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from passage_eval.io import pinned

SCHEMA = "passage-candidate-replicas/v1"
REPLICA_SEEDS = (1234, 3407)


class ReplicaCatalogError(ValueError):
    """The catalogue is unsealed, mismatched or does not freeze the requested replica."""


def validate_replica_catalog(
    catalog: Any, *, final_candidates: Mapping[str, Any], final_candidates_sha256: str
) -> dict:
    """Check identities only; these hashes cannot establish how weights were trained or executed."""

    def require(condition, message):
        if not condition:
            raise ReplicaCatalogError(message)

    require(isinstance(catalog, Mapping), "replica catalogue must be an object")
    require(
        set(catalog) == {"schema", "status", "final_candidates_sha256", "final_decoders_sha256", "replicas"},
        "replica catalogue keys differ",
    )
    require(catalog["schema"] == SCHEMA and catalog["status"] == "sealed", "replica catalogue is not sealed")
    require(
        catalog["final_candidates_sha256"] == final_candidates_sha256,
        "replica catalogue names another frozen 2026 manifest",
    )
    require(
        catalog["final_decoders_sha256"] == final_candidates["reference_decoders"]["module_sha256"],
        "replica catalogue names another decoder",
    )
    replicas = catalog["replicas"]
    require(
        isinstance(replicas, Mapping) and set(replicas) == {str(seed) for seed in REPLICA_SEEDS},
        "replica catalogue must declare 1234 and 3407",
    )
    legacy = {final_candidates["candidates"][name]["checkpoint_256"]["sha256"] for name in ("seg", "reg")}
    seen = set()
    for seed in REPLICA_SEEDS:
        entry = replicas[str(seed)]
        require(
            isinstance(entry, Mapping) and set(entry) == {"status", "seg_checkpoint_sha256", "reg_checkpoint_sha256"},
            f"replica seed {seed} keys differ",
        )
        require(entry["status"] in ("pending", "sealed"), f"replica seed {seed} status differs")
        for name in ("seg", "reg"):
            digest = entry[f"{name}_checkpoint_sha256"]
            if entry["status"] == "pending":
                require(digest is None, f"pending seed {seed} must not claim a checkpoint")
            else:
                require(pinned.is_hex64(digest), f"seed {seed}/{name} requires an actual frozen checkpoint hash")
                require(
                    digest not in legacy and digest not in seen,
                    "a replica cannot relabel 2026 weights or another replica",
                )
                seen.add(digest)
    return copy.deepcopy(dict(catalog))


def replica_identity(catalog: Mapping[str, Any], procedure: str, seed: int) -> dict[str, Any]:
    """Select one identity from an already validated catalogue; no pending entry is usable."""
    if procedure not in ("seg", "reg") or type(seed) is not int or seed not in REPLICA_SEEDS:
        raise ReplicaCatalogError("a replica is Seg or Reg with training seed 1234 or 3407")
    entry = catalog["replicas"][str(seed)]
    if entry["status"] != "sealed":
        raise ReplicaCatalogError(f"training seed {seed} checkpoints are still pending")
    return {"procedure": procedure, "seed": seed, "checkpoint_sha256": entry[f"{procedure}_checkpoint_sha256"]}


def load_replica_catalog(path: str | Path, sha256: str, *, harness_commit: str | None = None) -> dict:
    """Host-side catalogue reader, pinned to the committed, unmodified 2026 manifest."""
    from passage_eval import cli
    from passage_eval.protocol.admission import committed_final_candidates

    frozen, frozen_sha = committed_final_candidates(harness_commit or cli._checkout_commit())
    catalog = pinned.read_pinned_json(path, sha256)
    return validate_replica_catalog(catalog, final_candidates=frozen, final_candidates_sha256=frozen_sha)
