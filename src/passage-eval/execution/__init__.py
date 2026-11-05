# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
