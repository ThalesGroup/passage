"""Complete ordinary-input requests and independent artifact rescoring."""

from passage_eval.execution.artifacts import RunArtifact, read_run_artifact, write_run_artifact
from passage_eval.execution.runner import (
    DecoderPlannerAdapter,
    LegacyPlannerAdapter,
    as_planner,
    run_requests,
    request_identity,
    PendingCompletionError,
    drain_pending_requests,
)
from passage_eval.execution.scoring import (
    NegativeRegretHalt,
    score_run_artifact,
    validate_qualified_reference,
    write_scored_artifact,
)

__all__ = [
    "RunArtifact",
    "read_run_artifact",
    "write_run_artifact",
    "run_requests",
    "score_run_artifact",
    "write_scored_artifact",
    "validate_qualified_reference",
    "NegativeRegretHalt",
    "request_identity",
    "DecoderPlannerAdapter",
    "LegacyPlannerAdapter",
    "as_planner",
    "PendingCompletionError",
    "drain_pending_requests",
]
