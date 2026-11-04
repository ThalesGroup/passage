# MIT License
#
# Copyright (c) 2026 THALES

"""Trusted execution chain: supervised, redacted, preemptible candidate workers.

This package is deliberately import-light: it only defines the identifiers that
are recorded in run manifests. The worker, supervisor and scorer-staging code
live in submodules so that importing one side never imports the other.

The supervisor (trusted) never imports candidate code. Candidate code runs only
inside ``python -I -m passage_eval.supervision.worker`` processes that receive
redacted requests over ``passage-ipc/v1``. See ``docs/benchmark/supervised-execution.md``.
"""

from __future__ import annotations

#: Framed wire protocol between supervisor and worker.
IPC_PROTOCOL = "passage-ipc/v1"

#: Versioned definition of what the supervised wall-clock measurement contains.
#: The in-process runner uses ``request_copy_through_validation_recost_completion``.
#: v3 adds the pass policy: a fresh worker and fresh tokens for every timed pass, and an untimed warm-up
#: on inputs disjoint from the roster. v2 artifacts (one worker for all passes) stay readable.
TIMING_BOUNDARY_ID = "passage-timing-boundary/v3-supervised-ipc-fresh-pass"
LEGACY_TIMING_BOUNDARY_IDS = ("passage-timing-boundary/v2-supervised-ipc",)
#: v4 (timing profile ``supervised_service_v1``): the wall budget and the scorer use ``service_wall_ms``; the
#: supervisor's validation and physical recost are recorded separately, outside the boundary.
SERVICE_TIMING_BOUNDARY_ID = "passage-timing-boundary/v4-supervised-ipc-service"
#: v5 (timing profile ``supervised_service_rewarm_v1``): as v4, and a worker restarted inside a warm pass is
#: warmed up again, outside the boundary, before the next measured request.
REWARM_TIMING_BOUNDARY_ID = "passage-timing-boundary/v5-supervised-ipc-service-rewarm"
READABLE_TIMING_BOUNDARY_IDS = LEGACY_TIMING_BOUNDARY_IDS + (
    TIMING_BOUNDARY_ID,
    SERVICE_TIMING_BOUNDARY_ID,
    REWARM_TIMING_BOUNDARY_ID,
)

#: Longest wait for a killed worker's process group to be reaped (``WorkerProcess.kill``).
WORKER_KILL_WAIT_S = 15.0

#: Why a run is made. ``verified-candidate-run`` is the only purpose that can lead to a verified result.
PURPOSE_DIAGNOSTIC = "diagnostic"
PURPOSE_VERIFIED = "verified-candidate-run"
PURPOSES = (PURPOSE_DIAGNOSTIC, PURPOSE_VERIFIED)

#: Modules a worker must never import (checked in ``sys.modules``; self-reported by the worker).
DEFAULT_FORBIDDEN_MODULES = ("onnxruntime",)

#: Redaction policies: what a candidate process may receive.
POLICY_ORDINARY_REQUEST = "ordinary-request/v1"
POLICY_DECODER_ONLY = "decoder-only/v1"

#: Manifest block written by the supervisor.
SUPERVISION_SCHEMA = "passage-supervision-v1"

__all__ = [
    "IPC_PROTOCOL",
    "TIMING_BOUNDARY_ID",
    "LEGACY_TIMING_BOUNDARY_IDS",
    "SERVICE_TIMING_BOUNDARY_ID",
    "REWARM_TIMING_BOUNDARY_ID",
    "READABLE_TIMING_BOUNDARY_IDS",
    "WORKER_KILL_WAIT_S",
    "PURPOSE_DIAGNOSTIC",
    "PURPOSE_VERIFIED",
    "PURPOSES",
    "DEFAULT_FORBIDDEN_MODULES",
    "POLICY_ORDINARY_REQUEST",
    "POLICY_DECODER_ONLY",
    "SUPERVISION_SCHEMA",
]
