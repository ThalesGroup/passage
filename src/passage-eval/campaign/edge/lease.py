"""Small, fail-closed helpers for a single CAVIAR-managed edge lease."""

from __future__ import annotations

import re
import secrets

from passage_eval.campaign.edge import EDGE_TARGETS

RESOURCE_PREFIX = "device:edge-"
_JOB_ID = re.compile(r"job:[0-9a-f]{16}\Z")


class EdgeLeaseError(ValueError):
    """A lease request or CAVIAR receipt is outside the narrow Edge contract."""


def validate_target(target: str) -> str:
    """Return one supported target, rejecting every alias and host fallback."""
    if target not in EDGE_TARGETS:
        raise ValueError("edge target must be Thor, Orin, or Tiger")
    return target


def resource_for(target: str) -> str:
    """Return the exact CAVIAR resource name for a supported target."""
    return RESOURCE_PREFIX + validate_target(target)


def holder_host_for(target: str) -> str:
    """Name the remote holder so CAVIAR retains uncertain nonzero jobs."""
    return "edge-" + validate_target(target)


def new_effect_key(target: str) -> str:
    """Make a one-use CAVIAR effect key for this individual attempt."""
    return f"passage-edge-{validate_target(target)}-{secrets.token_hex(16)}"


def parse_job_id(value: str) -> str:
    """Parse the sole durable receipt emitted by ``cavops job-run --wait``."""
    job_id = value.strip()
    if not _JOB_ID.fullmatch(job_id):
        raise EdgeLeaseError("cavops job receipt must be one job:<16 lowercase hex> identifier")
    return job_id


__all__ = [
    "EdgeLeaseError",
    "holder_host_for",
    "new_effect_key",
    "parse_job_id",
    "resource_for",
    "validate_target",
]
