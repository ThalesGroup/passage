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

"""Versioned campaign tooling: roster partition, sealed plan, batch runner and aggregator.

A *campaign* runs every procedure of a sealed plan over an admissible roster, split into batches (lots)
of at most 1,024 tasks run in lease or timeout windows, on the host. Five modules, each with a command
(``python -I -m passage_eval.campaign`` with ``partition``, ``verify-partition``, ``check-plan``, ``run``,
``relaunch-batch``, ``aggregate`` and ``rehearsal-receipt``):

* :mod:`passage_eval.campaign.partition` splits the admissible roster into contiguous child rosters derived with
  ``io.roster.derive_roster``: every child keeps the parent's ranks and its geographic blocks.
* :mod:`passage_eval.campaign.plan` reads a SHA-256-pinned plan (``passage-campaign-plan/v1``) with strict unknown-key
  refusal. The plan seals the split, the roster digests, the reference policy and admission chain, the budget, the
  timing profile and every supervision limit, the warm-up roster, the paired order, the window of dynamic admission
  with its measured overheads and safety margin (checked with ``protocol.batching.admission_sizing``), the
  procedures with their decoder, tree and environment pins and candidate identities, the harness source hash and
  commit, the aggregation rule, the seeds, the target and the protocol record digest.
* :mod:`passage_eval.campaign.runner` executes the batches in lease windows with a lock and intent, completion and
  failure receipts. Every session admits its requests dynamically and defers what its window cannot hold to the next
  window. A killed, invalid or interrupted attempt is relaunched automatically under the judge's R7 rule (at most
  twice, then split; a single task killed twice is a candidate failure; more than 5% of batches relaunched stops the
  campaign); ``relaunch-batch`` retries an attempt stopped by a harness error.
* :mod:`passage_eval.campaign.aggregate` recomputes the six metrics from concatenated per-task rows (never an
  average of batch quantiles), with the parent geographic blocks, the source, excluded and admissible counts,
  failures kept with the canonical penalty, a censoring-aware warm latency block, the cold start per session and the
  relaunch counts.
* :mod:`passage_eval.campaign.rehearsal` builds the ``passage-native-chain-rehearsal/v1`` receipt of a validation
  rehearsal from its real campaign outputs.

The test split needs an all-GREEN ``passage-test-admission/v1`` record (``protocol.admission``) and an explicit
``--allow-test-split``; it is refused before any file of the split is read. Only the host target is implemented:
an edge target is refused with ``EdgeTransportNotQualified``. Hashes prove byte correspondence, not authorship.
"""

from __future__ import annotations

PARTITION_SCHEMA = "passage-campaign-partition/v1"
PLAN_SCHEMA = "passage-campaign-plan/v1"
COMPLETION_SCHEMA = "passage-campaign-batch-completion/v1"
INTENT_SCHEMA = "passage-campaign-batch-intent/v1"
FAILURE_SCHEMA = "passage-campaign-batch-failure/v1"
STATUS_SCHEMA = "passage-campaign-status/v1"
RELAUNCH_SCHEMA = "passage-campaign-relaunch/v1"
SPLIT_SCHEMA = "passage-campaign-r7-split/v1"
IMPUTED_SCHEMA = "passage-campaign-imputed-failure/v1"
AGGREGATE_SCHEMA = "passage-campaign-aggregate/v1"
REHEARSAL_RECEIPT_SCHEMA = "passage-native-chain-rehearsal/v1"
EDGE_VALIDATION_RELAY_SCHEMA = "passage-edge-validation-relay/v1"


class EdgeTransportNotQualified(PermissionError):
    """An edge target was requested; no qualified remote transport exists in this harness."""


__all__ = [
    "AGGREGATE_SCHEMA",
    "COMPLETION_SCHEMA",
    "EDGE_VALIDATION_RELAY_SCHEMA",
    "EdgeTransportNotQualified",
    "FAILURE_SCHEMA",
    "IMPUTED_SCHEMA",
    "INTENT_SCHEMA",
    "PARTITION_SCHEMA",
    "PLAN_SCHEMA",
    "REHEARSAL_RECEIPT_SCHEMA",
    "RELAUNCH_SCHEMA",
    "SPLIT_SCHEMA",
    "STATUS_SCHEMA",
]
