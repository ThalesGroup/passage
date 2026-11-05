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

"""Test admission gate ``passage-test-admission/v1``: a record and evidence verifier.

A *record* states the sealed protocol decisions, the pinned harness, the candidate identities and the evidence
files that support six gates. ``check_admission`` evaluates every gate and returns ``GREEN`` or ``RED`` per gate,
with machine-readable reasons (``{"code", "detail"}``). It never raises for a ``RED`` gate: an unreadable or
malformed record turns every gate ``RED``. The test split may be opened only when all six gates are ``GREEN``.

* **G1 protocol sealed.** The record's status is ``sealed``; the sealed protocol document matches its SHA-256 and
  carries no ``proposed_not_sealed`` marker; every decision of ``DECISIONS`` is present, valid and ``sealed``, with no
  ``draft``/``proposed_not_sealed``/``todo`` value left; an explicit approval names the approver, the time and the
  SHA-256 of the approved record (``record_core_sha256``: the record without its approval). The authorised harness
  commit is the record's commit.
* **G2 native chain validated on validation.** The running harness is the record's commit with a clean working
  tree (no uncommitted change under ``HARNESS_TRACKED_PATHS``) and the record's source hash. A
  ``passage-native-chain-rehearsal/v1`` receipt of a complete validation rehearsal on the host, made with that
  harness, with the frozen candidates, the sealed budget and timing profile, a performed warm-up, an exercised
  re-warm, a deliberately killed batch relaunched under the planned R7 rule (``deliberate_kill``), an exercised resume
  without any unplanned relaunch, and one ``passage_eval.campaign`` aggregate per sealed
  procedure. Each aggregate is bound to its procedure, the host, the rehearsal plan, the sealed budget, timing
  profile and seeds; for every sealed seed it equals the canonical scorer and has at least the sealed minimum of
  warm requests. No two procedures share an aggregate, and the receipt and aggregates carry no marker of synthetic,
  simulated or unsealed content.
* **G3 edge leases per target.** For every edge target of the sealed targets (``thor``, ``orin``, ``tiger``), a
  ``passage-edge-lease-evidence/v1`` file of a real ``cavops``-leased remote run: target identity attestation, lease
  activity and job ids, a continuous lease verification log covering the run, a drain confirmed before release, and
  transport-loss and cancellation test receipts. Absent evidence is ``RED``. Evidence that declares itself
  synthetic, loopback, simulated, mocked, self-generated or unqualified is ``RED``. G3 does not require an edge
  target that the sealed ``targets`` decision leaves out; the report lists, per edge target, whether an edge
  latency claim is admitted (``edge_latency_claims``).
* **G4 references and roster sealed.** The source roster (full population, ``passage-roster-v2``, complete block
  table bound to the installed block rule) of the sealed dataset revision, dataset root, evaluation cell and
  population (row count and id inventory digest), with the sealed block rule and degrees and, when sealed, the
  unplaced block for tasks without footprint; the reference admission policy, inventory, admissible roster and chain
  (re-derived here), matching the sealed reference policy, cohort, search limit, category table and exclusion caps.
  The admissible roster's block adequacy under the sealed criterion is reported (``admitted['block_adequacy']`` and
  ``admitted['block_analysis']``), not gated: a failed criterion makes the block analysis descriptive only, and
  unplaced tasks above the sealed ``max_unplaced_fraction`` stop the block analysis for diagnosis (judge R5, status
  ``halted_unplaced_cap``) without stopping the run or its descriptive regret and latency.
* **G5 warm-up and latency tested.** A validation warm-up roster disjoint from the admissible roster, built with the
  sealed salt and size on the admissible roster's dataset root, revision and cell; the rehearsal latency blocks meet
  the sealed minimum; the sealed censoring rule is the installed one.
* **G6 artifacts and candidate identities.** ``docs/benchmark/final-candidates.json`` is read as committed at the
  record's harness commit; its checkpoint SHA-256 values equal the record's; the installed ``final_decoders.py`` and
  the file's decoder pin both have the frozen SHA-256 ``FROZEN_FINAL_DECODERS_SHA256``; every declared target has a
  bundle file whose digest is the declared bundle digest and whose lineage names the checkpoint, and every edge
  target its engine and container digests; the decoder tree pin equals the tree digest of the installed decoder
  package. A grant re-hashes ``final-candidates.json`` and ``final_decoders.py`` before each use.

What the verifier cannot do. Every check is structural, numerical or a hash comparison: a hash proves byte
correspondence, not authorship. Evidence that is complete, consistent and free of the explicit markers is accepted;
the approval in G1 binds the evidence digests to the approver's statement, and the approver remains responsible for
their authenticity. The verifier never contacts a target, a lease registry or a network service.

Paths in a record are relative to the evidence directory (default: the record's directory); absolute paths,
``..`` and symbolic links are refused.

**Preparation grant.** G4 and G5 need artefacts that only reading the test split produces. ``grant_test_preparation``
returns a ``PreparationGrant`` when the record's only RED reasons are those artefacts (the five reference files of
G4, the two warm-up files of G5, and the ``population`` decision that the roster builder inventories, while it is
still ``required`` and null), G1, G2, G3 and G6 being otherwise GREEN, and a separate
``passage-test-preparation-approval/v1`` file of the author names the record's SHA-256, the preparation readers it
allows (``guards.PREPARATION_READERS``) and an expiry at most ``PREPARATION_MAX_WINDOW`` after the approval. The
guards accept it only for those readers (roster build, block census, reference inventory, warm-up roster); a run,
a candidate and scoring still need an all-GREEN ``AdmissionGrant``.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import importlib.util
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from passage_eval.io import pinned
from passage_eval.protocol.geographic_blocks import BLOCK_RULES, DEFAULT_BLOCK_RULE

SCHEMA = "passage-test-admission/v1"
STRICT_PROFILE = "strict"
DESCRIPTIVE_NATIVE_PROFILE = "descriptive_native_2026"
DESCRIPTIVE_SINGLE_SEED_PROFILE = "descriptive_native_single_seed"
DESCRIPTIVE_PROFILES = (DESCRIPTIVE_NATIVE_PROFILE, DESCRIPTIVE_SINGLE_SEED_PROFILE)
RULE8_NATIVE_PROFILE = "rule8_native_persistent"
# Only these explicit profiles accept the non-supervised timing boundary; legacy strict stays unchanged.
NATIVE_PROFILES = (*DESCRIPTIVE_PROFILES, RULE8_NATIVE_PROFILE)
REPLICA_PROFILES = (DESCRIPTIVE_SINGLE_SEED_PROFILE, RULE8_NATIVE_PROFILE)
PLANNED_SEEDS = (1234, 2026, 3407)
DESCRIPTIVE_ANALYSIS = {"mode": "descriptive_only"}
# This is a roster-content pin, not a digest of selected ids or of its JSON file bytes.
# Its origin is the already recorded public-test roster identity, not a new test read.
HISTORICAL_S_ROSTER_SHA256 = "0c1619faee0d743adf9c486d49e11f33c8a0d2517558de39a026cd3d5fec713c"
DESCRIPTIVE_DATASET = {
    "repo": "thalesgroup/passage",
    "commit": "c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae",
    "config": "256x256",
    "resolution": 256,
}
DESCRIPTIVE_ROSTER_SALT = "passage-public-test-v1/r0256/roster-v1/"
NATIVE_VALIDATION_SCHEMA = "passage-native-persistent-validation/v1"
NATIVE_INCOMPLETE_POLICY = {"mode": "abort_incomplete_session_no_automatic_replay"}
CHECK_SCHEMA = "passage-test-admission-check/v1"
REHEARSAL_SCHEMA = "passage-native-chain-rehearsal/v1"
EDGE_EVIDENCE_SCHEMA = "passage-edge-lease-evidence/v1"
FAULT_TEST_SCHEMA = "passage-edge-fault-test/v1"
LEASE_CHECK_SCHEMA = "passage-edge-lease-check/v1"
AGGREGATE_SCHEMA = "passage-campaign-aggregate/v1"

GREEN = "GREEN"
RED = "RED"
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
GATE_TITLES = {
    "G1": "protocol sealed and approved",
    "G2": "native chain validated on validation",
    "G3": "edge leases integrated per target",
    "G4": "references and roster sealed",
    "G5": "warm-up and latency tested",
    "G6": "artifacts and candidate identities verified",
}
HOST = "host"
EDGE_TARGETS = ("thor", "orin", "tiger")
TARGETS = (HOST, *EDGE_TARGETS)
CANDIDATES = ("seg", "reg")
SEALED = "sealed"
REQUIRED = "required"
DECISION_STATUSES = (SEALED, "proposed_not_sealed", REQUIRED)
#: The label-free selection key that defines roster S.  Keep this text equal
#: to ``scripts/build_public_test_roster.py``'s recorded selection key.
ROSTER_SELECTION_RULE = "sha256(utf-8(salt + id)) hexdigest, ascending lexicographic"
S_ROSTER_SIZE = 1024
#: Decoder entries that run the frozen kernels of ``final_decoders.py`` without changing them.
FROZEN_DECODER_ENTRIES = {
    "seg": (
        "passage_eval.baselines.final_bounded:SegFinalBounded",
        "passage_eval.baselines.final_operator_adapters:SegFinalDecoder",
    ),
    "reg": (
        "passage_eval.baselines.final_bounded:RegFinalBounded",
        "passage_eval.baselines.final_operator_adapters:RegFinalDecoder",
    ),
}
#: SHA-256 of the frozen ``src/passage-eval/baselines/final_decoders.py`` (``reference_decoders.module_sha256`` of
#: ``docs/benchmark/final-candidates.json``). The installed module and the frozen file must both name it.
FROZEN_FINAL_DECODERS_SHA256 = "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
#: The frozen candidate file, relative to the repository root; G6 reads it as committed at the record's commit.
FINAL_CANDIDATES_RELATIVE = "docs/benchmark/final-candidates.json"
#: Planner entries an admitted test plan may run as its ``astar`` procedure. They live in the package of the frozen
#: decoders, so the record's decoder tree pin (G6) is also their planner tree pin.
CANONICAL_PLANNER_ENTRIES = ("passage_eval.baselines.classical:CanonicalAStarPlanner",)
#: Largest sealed batch (lot) size; equal to ``campaign.partition.MAX_BATCH``.
MAX_BATCH_SIZE = 1024
#: Repository paths whose uncommitted changes (tracked or untracked) make the running harness differ from its
#: commit: G2 requires none.
HARNESS_TRACKED_PATHS = ("src", "scripts", FINAL_CANDIDATES_RELATIVE, "pyproject.toml", "uv.lock")

#: Every decision the protocol author must seal before the test split is opened, with the open decision it closes
#: (``harness-gaps`` section 13 numbering, track decisions D1-D8, transport gates P1-P2).
DECISIONS: dict[str, str] = {
    "dataset_revision": "{repo, commit (40 hex), config, resolution} of the retained published dataset revision (protocol section 1)",
    "dataset_root_sha256": "SHA-256 of the pinned dataset root receipt (dataset_root_identity) of that revision",
    "evaluation_cell": "{cost_model, cost_weight, task_variant} of the evaluated cell (protocol section 1)",
    "roster_selection": (
        "{rule, salt, n} of roster S: the label-free SHA-256 ordering, its sealed salt and exactly 1024 tasks"
    ),
    "population": "{n_rows, ids_sha256} of the selected source roster S (not selection.population's full split inventory), inventoried before the run",
    "budget": "per-request budget {wall_time_ms, max_expansions, max_queue_operations, max_live_states} (decision 1, D6)",
    "timing_profile": "supervised_service_v1, supervised_service_rewarm_v1, or native_raw_path_persistent_v1 for an explicit native profile (D1)",
    "timing_boundary_id": "timing boundary identifier recorded by that profile (D1)",
    "deadline_grace_ms": "grace after the wall deadline before preemption (decision 1, D6)",
    "hard_timeout_ms": "hard timeout of a request without a wall budget (D4)",
    "startup_timeout_s": "worker start timeout (D4)",
    "max_restarts": "restart cap per seed (D2, D4)",
    "worker_rlimits": "worker resource limits {nofile, fsize_mb, as_mb, cpu_s} (D4)",
    "reference_policy": "released_labels or independent_analytic_reference (decision 2)",
    "reference_cohort_id": "named cohort of the independent policy; null for released_labels (decision 2)",
    "reference_max_expansions": "exact reference search limit, at least N*N; 65536 and 65537 give other ids (decision 2)",
    "reference_category_table_sha256": "SHA-256 of the sealed exclusion/halt category table (decisions 3, 10)",
    "reference_exclusion_caps": (
        "{max_total_fraction, max_block_fraction, min_block_exclusions} of the reference inventory (judge R1)"
    ),
    "block_analysis": (
        "{rule, block_degrees, missing_footprint, min_blocks, max_block_share, max_unplaced_fraction}: block rule "
        "and adequacy criterion sealed before the label-free census (judge R5, tightening 1)"
    ),
    "censoring_rule": "latency censoring rule {id} used for percentiles (decision 4, D3)",
    "min_warm_requests": "minimum warm requests of a latency row (manuscript: 100)",
    "cold_start_policy": "what is reported of the cold start (decision 5)",
    "restart_and_failed_warmup_policy": "treatment of a restart and of a failed warm-up (decision 6, D2, D8)",
    "paired_order": "{granularity, procedures, schedule_seed} of the paired order (decision 7)",
    "batching": "{batch_size (1..1024), lease_or_timeout_ms, fixed_overhead_ms, per_seed_overhead_ms, "
    "per_request_overhead_ms, persistence_ms, observed_request_ms {procedure: ms}, safety_margin (>= 3)}: measured "
    "overheads and window of dynamic request admission (decision 8, D4, R7)",
    "killed_batch_policy": "a killed batch: imputed failures or ambiguous case (decision 8)",
    "candidate_start_failure_policy": "a candidate that cannot start: candidate failure or infrastructure (decision 9)",
    "missing_footprint_policy": "a task without footprint: halt of block analysis or declared exclusion (decision 10)",
    "warmup": "{salt, k, dataset_root_sha256 (the sealed dataset root)} of the warm-up roster (decision 11)",
    "infrastructure_relaunch_policy": "traced relaunch policy of a batch failed by infrastructure (decision 12)",
    "r7_policy": "{max_ambiguous_attempts, single_task_candidate_failures, relaunch_stop_fraction, candidate_attributable}: executable R7 relaunch decisions",
    "harness_commit_authorization": "{commit} of the harness authorised and committed on GitLab (decision 13)",
    "targets": "targets of the campaign among host, thor, orin, tiger",
    "edge_timing_boundary": "timing boundary of an edge row: network included or on-target timing (P1)",
    "infrastructure_failure_policy": "transport or lease loss: batch invalidated, never a candidate penalty (P2)",
    "seeds": "training seeds evaluated (protocol: 2026 preliminary; 1234, 2026, 3407 confirmatory)",
    "strict_comparison": "whether protocol comparisons require identical recorded limits (D5)",
    "confirmatory_analysis": (
        "{a, b, target, seeds, alpha, draws, permutations, bootstrap_seed, permutation_seed, latency_bootstrap_seed}: "
        "confirmatory family (a = seg, b = reg), its procedure target and seeds, and the resampling of "
        "campaign.statistics, sealed identically in every seed's record before the first test pass (judge R8)"
    ),
}
_POLICY_TEXT_DECISIONS = (
    "cold_start_policy",
    "restart_and_failed_warmup_policy",
    "killed_batch_policy",
    "candidate_start_failure_policy",
    "missing_footprint_policy",
    "infrastructure_relaunch_policy",
    "edge_timing_boundary",
    "infrastructure_failure_policy",
)
_RECORD_KEYS = {"schema", "status", "split", "record_id", "protocol", "harness", "candidates", "evidence", "approval"}
_OPTIONAL_RECORD_KEYS = {"notes"}
_PROTOCOL_KEYS = {"version", "document", "decisions"}
_HARNESS_KEYS = {"commit", "source_sha256"}
_CANDIDATE_KEYS = {
    "checkpoint_sha256",
    "predictor",
    "decoder",
    "decoder_tree_sha256",
    "final_decoders_sha256",
    "targets",
}
_TARGET_KEYS = {"bundle", "bundle_sha256", "engine_sha256", "container_digest"}
_EVIDENCE_KEYS = {"native_chain_rehearsal", "edge", "references", "warmup"}
_REFERENCE_KEYS = ("source_roster", "policy", "inventory", "admissible_roster", "chain")
_EXCLUSION_CAP_KEYS = ("max_total_fraction", "max_block_fraction", "min_block_exclusions")
_BLOCK_ANALYSIS_KEYS = {
    "rule",
    "block_degrees",
    "missing_footprint",
    "min_blocks",
    "max_block_share",
    "max_unplaced_fraction",
}
_CONFIRMATORY_KEYS = {
    "a",
    "b",
    "target",
    "seeds",
    "alpha",
    "draws",
    "permutations",
    "bootstrap_seed",
    "permutation_seed",
    "latency_bootstrap_seed",
}
_WARMUP_KEYS = ("roster", "selection")
_CELL_KEYS = ("cost_model", "cost_weight", "task_variant")
_ROSTER_SELECTION_KEYS = ("rule", "salt", "n")
#: Aggregate problems that leave the rehearsal latency of some sealed seed unshown (G5 follows them).
_LATENCY_UNSHOWN = frozenset(
    {
        "aggregate_schema_invalid",
        "aggregate_seeds_differ",
        "aggregate_latency_missing",
        "aggregate_latency_below_minimum",
    }
)
_APPROVAL_KEYS = {"approver", "approved_at", "approved_record_sha256"}

_UNSEALED_PATTERNS = (
    re.compile(r"proposed_not_sealed", re.IGNORECASE),
    re.compile(r"\bdraft\b", re.IGNORECASE),
    re.compile(r"\bunsealed\b", re.IGNORECASE),
    re.compile(r"\b(todo|tbd|placeholder)\b", re.IGNORECASE),
)
#: Markers of evidence that does not come from a real leased run on a target.
NON_EVIDENCE_MARKERS = (
    "synthetic",
    "loopback",
    "simulated",
    "simulation",
    "fake",
    "mock",
    "stub",
    "dummy",
    "self-generated",
    "self_generated",
    "selfgenerated",
    "localhost",
    "127.0.0.1",
    "unqualified",
    "not_qualified",
    "template",
)
_ACTIVITY_ID = re.compile(r"job:[0-9a-f]{16}\Z")
_DIGEST_TEXT = re.compile(r"(?:[0-9a-f]{40,}|(?:sha256|SHA256):[0-9A-Za-z+/]{40,}={0,2})\Z")
_CONTAINER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
#: Largest accepted gap between two lease checks, in multiples of the declared period.
LEASE_GAP_TOLERANCE = 2.0


class AdmissionRecordError(ValueError):
    """A record whose structure cannot be evaluated."""


# ----------------------------------------------------------------------------------------------- results


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    reasons: tuple[dict[str, str], ...] = ()

    @property
    def title(self) -> str:
        return GATE_TITLES[self.gate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "title": self.title,
            "status": self.status,
            "reasons": [dict(r) for r in self.reasons],
        }


@dataclass(frozen=True)
class AdmissionReport:
    record_name: str
    record_sha256: str | None
    record_core_sha256: str | None
    evidence_dir: str | None
    split: str | None
    gates: tuple[GateResult, ...]
    file_pins: tuple[tuple[str, str], ...] = ()
    admitted: Mapping[str, Any] = field(default_factory=dict)
    #: Absolute files outside the evidence directory that the verdict depends on (``final-candidates.json`` and the
    #: installed ``final_decoders.py``), with their SHA-256 at check time.
    external_pins: tuple[tuple[str, str], ...] = ()

    @property
    def all_green(self) -> bool:
        return len(self.gates) == len(GATES) and all(gate.status == GREEN for gate in self.gates)

    @property
    def edge_latency_claims(self) -> dict[str, str]:
        """Per edge target, whether this record admits an edge latency claim (sealed target and all six GREEN)."""
        targets = self.admitted.get("targets") or []
        return {
            target: "admitted"
            if self.all_green and target in targets
            else ("not_admitted: not a sealed target" if target not in targets else "not_admitted: gates RED")
            for target in EDGE_TARGETS
        }

    def gate(self, name: str) -> GateResult:
        return next(gate for gate in self.gates if gate.gate == name)

    def red_gates(self) -> list[str]:
        return [gate.gate for gate in self.gates if gate.status != GREEN]

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": CHECK_SCHEMA,
            "admission_schema": SCHEMA,
            "record": self.record_name,
            "record_sha256": self.record_sha256,
            "record_core_sha256": self.record_core_sha256,
            "evidence_dir": self.evidence_dir,
            "split": self.split,
            "verdict": GREEN if self.all_green else RED,
            "all_green": self.all_green,
            "gates": [gate.to_dict() for gate in self.gates],
            "test_split": "may be opened" if self.all_green else "refused",
            "edge_latency_claims": self.edge_latency_claims,
            "limits": "structural, numerical and hash checks; a hash does not authenticate who produced a file",
        }
        if self.admitted.get("profile") in DESCRIPTIVE_PROFILES:
            result["claim_scope"] = {
                "profile": self.admitted["profile"],
                "confirmatory_allowed": False,
                "reproducible_entry": False,
            }
        elif self.admitted.get("profile") == RULE8_NATIVE_PROFILE:
            result["claim_scope"] = {
                "profile": RULE8_NATIVE_PROFILE,
                "stage": "admission_only",
                "reproducible_entry": False,
                "requires": "three-seed paired CIs, complete row sidecars, published source and a locally rebuildable image",
            }
        return result


def format_report(report: AdmissionReport) -> str:
    """The gate table printed by ``passage-eval admission-check``."""
    lines = [
        f"{SCHEMA} admission check",
        f"record:        {report.record_name}",
        f"record sha256: {report.record_sha256 or 'unreadable'}",
        f"evidence dir:  {report.evidence_dir or 'none'}",
        "",
        f"{'Gate':<5} {'Status':<6} {'Title':<45} Reasons",
        f"{'-' * 5} {'-' * 6} {'-' * 45} {'-' * 40}",
    ]
    for gate in report.gates:
        reasons = list(gate.reasons) or [{"code": "ok", "detail": ""}]
        for index, reason in enumerate(reasons):
            text = reason["code"] + (f": {reason['detail']}" if reason.get("detail") else "")
            if index == 0:
                lines.append(f"{gate.gate:<5} {gate.status:<6} {gate.title:<45} {text}")
            else:
                lines.append(f"{'':<5} {'':<6} {'':<45} {text}")
    green = sum(gate.status == GREEN for gate in report.gates)
    verdict = "GREEN: the test split may be opened" if report.all_green else "RED: the test split stays refused"
    claims = report.edge_latency_claims
    admitted = [target for target, claim in claims.items() if claim == "admitted"]
    refused = [target for target, claim in claims.items() if claim != "admitted"]
    lines += [
        "",
        f"Result: {verdict} ({green}/{len(GATES)} gates GREEN)",
        f"Edge latency claims admitted: {', '.join(admitted) or 'none'}; "
        f"not admitted: {', '.join(refused) or 'none'} (no edge latency claim may be reported for them)",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- helpers


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def parse_timestamp(value: Any) -> _dt.datetime:
    """An ISO 8601 timestamp with an explicit time zone."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a timestamp string is required")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = _dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no time zone")
    return parsed


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and abs(value) != float("inf")
    )


def _strings(value: Any) -> list[str]:
    """Every string, key or value, inside a JSON value."""
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_strings(item))
    return found


def unsealed_markers(value: Any) -> list[str]:
    """The unsealed markers (``draft``, ``proposed_not_sealed``, ``todo``...) found in a JSON value."""
    hits = []
    for text in _strings(value):
        for pattern in _UNSEALED_PATTERNS:
            match = pattern.search(text)
            if match:
                hits.append(match.group(0).lower())
    return sorted(set(hits))


def non_evidence_markers(value: Any) -> list[str]:
    """Markers of synthetic, loopback or self-generated evidence found in a JSON value (case-insensitive).

    Digest strings (hexadecimal, or ``sha256:``/``SHA256:`` followed by a hexadecimal or base64 digest) are not
    words and are skipped, so that a digest cannot spell a marker by chance.
    """
    hits = set()
    for text in _strings(value):
        if _DIGEST_TEXT.fullmatch(text):
            continue
        lowered = text.lower()
        hits.update(marker for marker in NON_EVIDENCE_MARKERS if marker in lowered)
    return sorted(hits)


def reconstruction_non_evidence_markers(value: Mapping[str, Any]) -> list[str]:
    """Treat the historical Docker tag as a label while checking every other marker."""
    checked = dict(value)
    command = value.get("build_command")
    if isinstance(command, list):
        checked["build_command"] = [
            re.sub("unqualified", "", arg, flags=re.IGNORECASE)
            if i > 0 and command[i - 1] in ("-t", "--tag") and isinstance(arg, str)
            else arg
            for i, arg in enumerate(command)
        ]
    return non_evidence_markers(checked)


def _flags_unqualified(value: Any) -> bool:
    """Whether a JSON value carries ``qualified: false`` or ``target_evidence: none`` anywhere."""
    if isinstance(value, Mapping):
        if value.get("qualified") is False or (
            "target_evidence" in value and value["target_evidence"] in ("none", None)
        ):
            return True
        return any(_flags_unqualified(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_flags_unqualified(item) for item in value)
    return False


def record_core_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical record without its ``approval`` block: the value an approval must name."""
    return pinned.canonical_sha256({key: value for key, value in record.items() if key != "approval"})


def authorization_scope_sha256(record: Mapping[str, Any]) -> str:
    """Pin the translated authorization's material scope, excluding generated preparation outputs."""
    scope = copy.deepcopy(dict(record))
    scope.pop("approval", None)
    scope.pop("notes", None)
    scope["protocol"]["decisions"].pop("population", None)
    scope["evidence"].pop("references", None)
    scope["evidence"].pop("warmup", None)
    scope["evidence"].pop("edge", None)
    scope["evidence"].pop("native_chain_rehearsal", None)
    return pinned.canonical_sha256(scope)


def _installed_final_decoders() -> Path:
    """The installed ``final_decoders.py``, located without importing it."""
    import passage_eval

    return Path(passage_eval.__file__).resolve().parent / "baselines" / "final_decoders.py"


def _repository_root() -> Path:
    """The checkout that provides the installed ``passage_eval`` (``src/passage-eval`` two levels below it)."""
    import passage_eval

    return Path(passage_eval.__file__).resolve().parents[2]


def _default_final_candidates() -> Path:
    return _repository_root() / FINAL_CANDIDATES_RELATIVE


def _running_harness() -> tuple[str | None, str]:
    from passage_eval import cli

    return cli._checkout_commit(), cli._source_hash()


def _git(root: Path, *arguments: str) -> bytes | None:
    """The standard output of one bounded ``git`` command, or ``None`` when it cannot run or fails."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _uncommitted_harness_paths(root: Path | None = None) -> list[str] | None:
    """Paths under ``HARNESS_TRACKED_PATHS`` with uncommitted changes, untracked files included (ignored files are
    not); ``None`` outside a git checkout."""
    output = _git(
        root or _repository_root(),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
        "--",
        *HARNESS_TRACKED_PATHS,
    )
    if output is None:
        return None
    entries = [entry for entry in output.decode("utf-8", errors="replace").split("\0") if entry]
    return sorted(entry[3:] if len(entry) > 3 and entry[2] == " " else entry for entry in entries)


def _committed_file(commit: str, relative: str, root: Path | None = None) -> bytes | None:
    """The bytes of ``relative`` at ``commit`` (40 hex), or ``None`` when git cannot show them."""
    if not pinned.is_hex40(commit):
        return None
    return _git(root or _repository_root(), "show", f"{commit}:{relative}")


def host_verifier_runtime_binding(
    harness: Mapping[str, Any], verifier_commit: str, *, verifier_source_sha256: str, root: Path | None = None
):
    """Verify a clean host-only admission change without relabelling the recorded runtime commit.

    The installed verifier remains independently identified in the report. Only a descendant whose tracked
    runtime, scripts, lockfiles and frozen candidate manifest differ in the host admission verifier alone is
    accepted; the recorded runtime source hash is recomputed from its Git tree.
    """
    root = root or _repository_root()
    base = harness.get("commit") if isinstance(harness, Mapping) else None
    if (
        not pinned.is_hex40(base)
        or not pinned.is_hex40(verifier_commit)
        or not pinned.is_hex64(harness.get("source_sha256"))
    ):
        return None
    if (
        _uncommitted_harness_paths(root) != []
        or _git(root, "merge-base", "--is-ancestor", base, verifier_commit) is None
    ):
        return None
    changed = _git(root, "diff", "--name-only", "-z", base, verifier_commit, "--", *HARNESS_TRACKED_PATHS)
    if changed is None or set(changed.decode("utf-8").split("\0")) - {"", "src/passage-eval/protocol/admission.py"}:
        return None

    def source_hash(commit):
        names = _git(root, "ls-tree", "-r", "--name-only", "-z", commit, "--", "src/passage", "src/passage-eval")
        if names is None:
            return None
        paths = names.decode("utf-8").split("\0")
        digest = hashlib.sha256()
        for package, prefix in (("passage", "src/passage/"), ("passage_eval", "src/passage-eval/")):
            selected = sorted(path for path in paths if path.startswith(prefix) and path.endswith(".py"))
            if not selected:
                return None
            for path in selected:
                content = _committed_file(commit, path, root)
                if content is None:
                    return None
                digest.update(f"{package}/{path[len(prefix) :]}".encode())
                digest.update(content)
        return digest.hexdigest()

    base_source = source_hash(base)
    if base_source != harness["source_sha256"] or source_hash(verifier_commit) != verifier_source_sha256:
        return None
    return {
        "runtime_commit": base,
        "runtime_source_sha256": base_source,
        "verifier_commit": verifier_commit,
        "changed_tracked_paths": [p for p in changed.decode("utf-8").split("\0") if p],
    }


def committed_final_candidates(commit: str, path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """The frozen candidate file and its SHA-256, refused unless it is the file committed at ``commit``.

    ``path`` defaults to ``docs/benchmark/final-candidates.json`` of the installed checkout. Raises ``ValueError``
    when the file is unreadable, differs from its committed bytes, or no longer pins ``FROZEN_FINAL_DECODERS_SHA256``.
    """
    path = Path(path) if path is not None else _default_final_candidates()
    try:
        raw = pinned.read_bounded(path)
        content = pinned.strict_json_loads(raw, name=path.name)
    except (pinned.PinnedFileError, OSError) as exc:
        raise ValueError(f"final-candidates.json is unavailable: {exc}") from exc
    committed = _committed_file(commit, FINAL_CANDIDATES_RELATIVE)
    if committed is None:
        raise ValueError(f"final-candidates.json cannot be read at commit {str(commit)[:12]!r}")
    if committed != raw:
        raise ValueError("final-candidates.json differs from the file committed at the harness commit")
    module_sha = ((content.get("reference_decoders") if isinstance(content, Mapping) else None) or {}).get(
        "module_sha256"
    )
    if module_sha != FROZEN_FINAL_DECODERS_SHA256:
        raise ValueError("final-candidates.json no longer pins the frozen final_decoders.py")
    return content, pinned.sha256_bytes(raw)


def committed_decoder_tree_sha256(commit: str, *, root: Path | None = None) -> str:
    """Hash the runtime decoder package as committed, using the ordinary tree algorithm."""
    import io
    import tarfile
    import tempfile
    from passage_eval.supervision import treehash

    if not pinned.is_hex40(commit):
        raise ValueError("invalid runtime commit")
    archive = _git(root or _repository_root(), "archive", commit, "src/passage-eval")
    if archive is None:
        raise ValueError("runtime decoder tree is unavailable")
    with tempfile.TemporaryDirectory(prefix="passage-runtime-tree-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(directory, filter="data")
        return treehash.hash_tree(
            Path(directory) / "src/passage-eval",
            role="decoder_entry",
            symlinks="refuse",
            strict=True,
            exclude_dirs=treehash.CANDIDATE_EXCLUDES,
        ).sha256


def decoder_tree_sha256(entry: str) -> str:
    """Tree digest of the top-level package of a ``module:attr`` decoder entry, as ``tree-hash`` prints it.

    The package is located without importing the decoder module.
    """
    from passage_eval.supervision import treehash

    module = entry.partition(":")[0]
    top = module.split(".")[0]
    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f"the decoder package {top!r} is not installed as a package")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise ValueError(f"the decoder package {top!r} has {len(locations)} locations")
    record = treehash.hash_tree(
        locations[0],
        role="decoder_entry",
        symlinks="refuse",
        strict=True,
        exclude_dirs=treehash.CANDIDATE_EXCLUDES,
    )
    return record.sha256


# ----------------------------------------------------------------------------------------------- structure


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionRecordError(message)


def _check_ref(value: Any, where: str, *, nullable: bool = True) -> None:
    if value is None:
        _require(nullable, f"{where} is required")
        return
    try:
        pinned.file_ref(value)
    except pinned.PinnedFileError as exc:
        raise AdmissionRecordError(f"{where}: {exc}") from exc


def validate_record_structure(record: Any) -> dict[str, Any]:
    """Check the record's keys and shapes (not its content); raise ``AdmissionRecordError``."""
    _require(isinstance(record, Mapping), "the record must be a JSON object")
    record = copy.deepcopy(dict(record))
    _require(record.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    keys = set(record)
    _require(
        _RECORD_KEYS <= keys <= _RECORD_KEYS | _OPTIONAL_RECORD_KEYS, f"record keys must be {sorted(_RECORD_KEYS)}"
    )
    _require(record["status"] in (SEALED, "draft"), "status must be 'sealed' or 'draft'")
    _require(record["split"] == "test", "an admission record admits the test split")
    _require(isinstance(record["record_id"], str) and record["record_id"].strip(), "record_id is required")
    _require("notes" not in record or isinstance(record["notes"], str), "notes must be text")
    protocol = record["protocol"]
    _require(
        isinstance(protocol, Mapping) and _PROTOCOL_KEYS <= set(protocol) <= _PROTOCOL_KEYS | {"profile"},
        f"protocol keys: {sorted(_PROTOCOL_KEYS)} with optional profile",
    )
    profile = protocol.get("profile", STRICT_PROFILE)
    _require(profile in (STRICT_PROFILE, *NATIVE_PROFILES), "unknown admission profile")
    descriptive = profile in DESCRIPTIVE_PROFILES
    _require(isinstance(protocol["version"], str), "protocol.version must be text")
    _check_ref(protocol["document"], "protocol.document")
    decisions = protocol["decisions"]
    _require(isinstance(decisions, Mapping), "protocol.decisions must be an object")
    _require(set(decisions) == set(DECISIONS), f"protocol.decisions must name exactly {sorted(DECISIONS)}")
    for name, entry in decisions.items():
        _require(
            isinstance(entry, Mapping) and {"value", "status"} <= set(entry) <= {"value", "status", "note"},
            f"decision {name} is {{value, status[, note]}}",
        )
        _require(entry["status"] in DECISION_STATUSES, f"decision {name} status must be one of {DECISION_STATUSES}")
    harness = record["harness"]
    _require(isinstance(harness, Mapping) and set(harness) == _HARNESS_KEYS, f"harness keys: {sorted(_HARNESS_KEYS)}")
    candidates = record["candidates"]
    _require(isinstance(candidates, Mapping) and set(candidates) == set(CANDIDATES), "candidates are seg and reg")
    for name, candidate in candidates.items():
        _require(isinstance(candidate, Mapping) and set(candidate) == _CANDIDATE_KEYS, f"candidate {name} keys")
        targets = candidate["targets"]
        _require(isinstance(targets, Mapping) and set(targets) <= set(TARGETS), f"candidate {name} targets")
        for target, entry in targets.items():
            target_keys = _TARGET_KEYS | ({"native_environment"} if descriptive else set())
            if profile == RULE8_NATIVE_PROFILE:
                target_keys |= {"container_reconstruction"}
                if isinstance(entry, Mapping) and "native_environment" in entry:
                    target_keys |= {"native_environment"}
            _require(isinstance(entry, Mapping) and set(entry) == target_keys, f"candidate {name} target {target}")
            _check_ref(entry["bundle"], f"candidate {name} target {target} bundle")
            if "native_environment" in target_keys:
                _check_ref(entry["native_environment"], f"candidate {name} target {target} native environment")
            if profile == RULE8_NATIVE_PROFILE:
                _check_ref(
                    entry["container_reconstruction"], f"candidate {name} target {target} container reconstruction"
                )
    evidence = record["evidence"]
    evidence_keys = _EVIDENCE_KEYS | ({"replica_catalog"} if profile in REPLICA_PROFILES else set())
    _require(
        isinstance(evidence, Mapping) and _EVIDENCE_KEYS <= set(evidence) <= evidence_keys,
        f"evidence keys: {sorted(evidence_keys)}",
    )
    if "replica_catalog" in evidence:
        _check_ref(evidence["replica_catalog"], "evidence.replica_catalog")
    _check_ref(evidence["native_chain_rehearsal"], "evidence.native_chain_rehearsal")
    _require(isinstance(evidence["edge"], Mapping) and set(evidence["edge"]) == set(EDGE_TARGETS), "evidence.edge")
    for target, ref in evidence["edge"].items():
        _check_ref(ref, f"evidence.edge.{target}")
    reference_keys = set(_REFERENCE_KEYS) | ({"historical_roster"} if profile in NATIVE_PROFILES else set())
    _require(
        isinstance(evidence["references"], Mapping) and set(evidence["references"]) == reference_keys,
        f"evidence.references keys: {list(_REFERENCE_KEYS)}",
    )
    for name, ref in evidence["references"].items():
        _check_ref(ref, f"evidence.references.{name}")
    _require(
        isinstance(evidence["warmup"], Mapping) and set(evidence["warmup"]) == set(_WARMUP_KEYS),
        f"evidence.warmup keys: {list(_WARMUP_KEYS)}",
    )
    for name, ref in evidence["warmup"].items():
        _check_ref(ref, f"evidence.warmup.{name}")
    approval = record["approval"]
    _require(
        approval is None
        or (isinstance(approval, Mapping) and set(approval) in (_APPROVAL_KEYS, {"authorization", "operator_receipt"})),
        "approval keys",
    )
    return record


# ----------------------------------------------------------------------------------------------- the checker


class _Gate:
    """Collects the reasons of one gate."""

    def __init__(self, name: str):
        self.name = name
        self.reasons: list[dict[str, str]] = []

    def red(self, code: str, detail: Any = "") -> None:
        self.reasons.append({"code": code, "detail": str(detail)[:400]})

    def result(self) -> GateResult:
        return GateResult(self.name, RED if self.reasons else GREEN, tuple(self.reasons))


def _descriptive_scope_problems(decisions: Mapping[str, Any], profile: str = DESCRIPTIVE_NATIVE_PROFILE) -> list[str]:
    """The authorised native campaign is S, Seg/Reg and one planned training seed per edge record."""
    values = {name: entry["value"] for name, entry in decisions.items()}
    exact = {
        "dataset_revision": DESCRIPTIVE_DATASET,
        "evaluation_cell": {"cost_model": "elevation", "cost_weight": 62.4, "task_variant": "obstacles"},
        "roster_selection": {"rule": ROSTER_SELECTION_RULE, "salt": DESCRIPTIVE_ROSTER_SALT, "n": S_ROSTER_SIZE},
        "reference_policy": "independent_analytic_reference",
        "reference_max_expansions": 65536,
    }
    problems = [
        f"{name} differs from the authorised descriptive S campaign"
        for name, value in exact.items()
        if values[name] != value
    ]
    seeds = values["seeds"]
    if not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int or seeds[0] not in PLANNED_SEEDS:
        problems.append("one explicitly sealed training seed among 1234, 2026, 3407 is required per record")
    elif profile == DESCRIPTIVE_NATIVE_PROFILE and seeds != [2026]:
        problems.append("the legacy 2026 profile is limited to seed 2026")
    targets = values["targets"]
    if not isinstance(targets, list) or not targets or any(target not in EDGE_TARGETS for target in targets):
        problems.append("targets must be a nonempty subset of thor, orin and tiger")
    order = values["paired_order"]
    procedures = order.get("procedures") if isinstance(order, Mapping) else None
    if (
        not isinstance(procedures, list)
        or len(procedures) != 2
        or not all(isinstance(procedure, str) for procedure in procedures)
        or set(procedures) != {"reg", "seg"}
    ):
        problems.append("the descriptive campaign runs exactly Seg and Reg")
    population = decisions["population"]
    if not (population["status"] == REQUIRED and population["value"] is None):
        if not isinstance(population["value"], Mapping) or population["value"].get("n_rows") != S_ROSTER_SIZE:
            problems.append("the source population is exactly the 1024 selected tasks of S")
    if not _is_int(values["min_warm_requests"]) or values["min_warm_requests"] < 100:
        problems.append("a descriptive latency row still requires at least 100 warm timed requests")
    from passage_eval.execution import timing

    if values["timing_profile"] == timing.NATIVE_PERSISTENT:
        if not isinstance(order, Mapping) or order.get("granularity") != "per_candidate":
            problems.append("persistent native candidates run sequentially, never interleaved by fragment")
        if values["max_restarts"] != 0 or values["r7_policy"] != NATIVE_INCOMPLETE_POLICY:
            problems.append("an incomplete native session is retained for diagnosis, never automatically replayed")
        warmup = values["warmup"]
        if not isinstance(warmup, Mapping) or warmup.get("k") != 1 or warmup.get("repeats") != 16:
            problems.append("native warm-up repeats one disjoint validation scene exactly 16 times initially")
    return problems


def _native_environment_problems(
    environment: Any, record: Mapping[str, Any], name: str, target: str, now: _dt.datetime
) -> list[str]:
    """Check a pinned operator attestation; this cannot observe device placement or authenticate its author."""
    candidate = record["candidates"][name]
    entry = candidate["targets"][target]
    expected = {
        "schema": "passage-native-environment/v1",
        "target": target,
        "candidate": name,
        "harness": record["harness"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "bundle_sha256": entry["bundle_sha256"],
        "engine_sha256": entry["engine_sha256"],
        **{key: candidate[key] for key in ("predictor", "decoder", "decoder_tree_sha256", "final_decoders_sha256")},
    }
    if not isinstance(environment, Mapping):
        return ["an operator attestation is required"]
    problems = [
        f"{key} differs from the admitted artifacts" for key, value in expected.items() if environment.get(key) != value
    ]
    if non_evidence_markers(environment) or _flags_unqualified(environment):
        problems.append("the attestation is marked as non-evidence")
    operator = environment.get("operator")
    if not isinstance(operator, str) or not operator.strip() or unsealed_markers(operator):
        problems.append("a named operator is required")
    try:
        if parse_timestamp(environment.get("observed_at")) > now:
            problems.append("the attestation is dated in the future")
    except (TypeError, ValueError):
        problems.append("observed_at must be a timestamp with a time zone")
    stack = environment.get("stack")
    keys = ("python", "platform", "accelerator", "driver", "runtime", "precision", "power_mode", "determinism")
    if not isinstance(stack, Mapping) or any(
        not isinstance(stack.get(key), str) or not stack[key].strip() for key in keys
    ):
        problems.append(
            "the native stack must name python, platform, accelerator, driver, runtime, precision, power mode and determinism"
        )
    elif stack["determinism"] not in ("bit-exact", "algorithmically deterministic", "non-deterministic"):
        problems.append("the determinism flag differs from the paper's declared flags")
    isolation = environment.get("isolation")
    if not isinstance(isolation, Mapping) or isolation.get("level") not in ("process", "bwrap"):
        problems.append("the applied process or bwrap isolation must be declared")
    elif not isinstance(isolation.get("security_boundary"), bool) or (
        isolation["level"] == "process" and isolation["security_boundary"] is not False
    ):
        problems.append("process isolation cannot be claimed as a security boundary")
    return problems


def approval_problems(record: Mapping[str, Any], *, now: _dt.datetime | None = None) -> list[dict[str, str]]:
    """Check either human approval or an operator binding to a recorded authorization."""
    now = now or _now()
    gate = _Gate("G1")
    approval = record["approval"]
    if approval is None:
        gate.red("approval_missing", "no explicit author approval")
    elif "authorization" in approval:
        authorization, receipt = approval["authorization"], approval["operator_receipt"]
        if not isinstance(authorization, Mapping) or set(authorization) != {
            "statement",
            "source",
            "approver",
            "approved_at",
            "scope_sha256",
        }:
            gate.red("authorization_invalid", "existing authorization provenance differs")
        else:
            if any(
                not isinstance(authorization[k], str) or not authorization[k].strip() for k in ("statement", "source")
            ):
                gate.red("authorization_invalid", "authorization statement and source are required")
            if authorization["approver"] is not None and (
                not isinstance(authorization["approver"], str) or not authorization["approver"].strip()
            ):
                gate.red("authorization_invalid", "approver must be text or unknown")
            if authorization["approved_at"] is not None:
                try:
                    if parse_timestamp(authorization["approved_at"]) > now:
                        gate.red("authorization_invalid", "authorization time is in the future")
                except (ValueError, TypeError):
                    gate.red("authorization_invalid", "authorization time is invalid")
            if authorization["scope_sha256"] != authorization_scope_sha256(record):
                gate.red("authorization_scope_mismatch", "material scope differs from the recorded authorization")
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "operator",
            "created_at",
            "record_core_sha256",
            "authorization_sha256",
        }:
            gate.red("operator_receipt_invalid", "operator receipt fields differ")
        else:
            if not isinstance(receipt["operator"], str) or not receipt["operator"].strip():
                gate.red("operator_receipt_invalid", "operator identity is required")
            try:
                if parse_timestamp(receipt["created_at"]) > now:
                    gate.red("operator_receipt_invalid", "operator receipt is in the future")
            except (ValueError, TypeError):
                gate.red("operator_receipt_invalid", "operator receipt time is invalid")
            if receipt["record_core_sha256"] != record_core_sha256(record):
                gate.red("operator_receipt_mismatch", "operator receipt does not bind current record")
            if not isinstance(authorization, Mapping) or receipt["authorization_sha256"] != pinned.canonical_sha256(
                authorization
            ):
                gate.red("operator_receipt_mismatch", "operator receipt does not bind authorization")
    else:
        approver = approval["approver"]
        if not isinstance(approver, str) or not approver.strip() or unsealed_markers(approver):
            gate.red("approval_approver_invalid", "the approver must be named")
        try:
            approved_at = parse_timestamp(approval["approved_at"])
            if approved_at > now:
                gate.red("approval_in_the_future", approval["approved_at"])
        except (ValueError, TypeError) as exc:
            gate.red("approval_timestamp_invalid", exc)
        if approval["approved_record_sha256"] != record_core_sha256(record):
            gate.red("approval_hash_mismatch", "the approval names another record (record changed after approval)")
    return gate.reasons


def native_input_projection(roster: Mapping[str, Any]) -> dict[str, Any]:
    """Label-free ordered input identities; no Parquet, reference or canonical-roster digest goes to a target."""
    from passage_eval.io import roster as roster_io

    checked = roster_io.validate_roster(roster)
    return {
        "schema": "passage-native-input-identities/v1",
        "tasks": [
            {
                "sample_id": task["sample_id"],
                "content_identity_sha256": roster_io.content_identity_digest(task["input_identity"]),
            }
            for task in checked["tasks"]
        ],
    }


def native_warmup_disjoint(warmup: Mapping[str, Any], measured: Mapping[str, Any]) -> bool:
    """Compare IDs and content identities of two already validated input rosters."""
    from passage_eval.io import roster as roster_io

    ids = {task["sample_id"] for task in measured["tasks"]}
    contents = {roster_io.content_identity_digest(task["input_identity"]) for task in measured["tasks"]}
    return all(
        task["sample_id"] not in ids and roster_io.content_identity_digest(task["input_identity"]) not in contents
        for task in warmup["tasks"]
    )


def native_causal_lease(observation, start, finish, drain, *, target: str, procedure: str) -> tuple[int, int]:
    """Validate recorded job events and drain facts without inventing exact lease timestamps."""
    from passage_eval.campaign.direct.direct_host_scoring import verify_drain
    from passage_eval.campaign.edge.lease import parse_job_id

    if not all(isinstance(item, Mapping) for item in (observation, start, finish, drain)):
        raise ValueError("causal evidence must contain four records")
    job = observation.get("job_id")
    if not isinstance(job, str) or parse_job_id(job) != job:
        raise ValueError("invalid CAVIAR job")
    activity = observation.get("activity")
    if (
        observation.get("schema") != "passage-cavops-job-observation/v1"
        or observation.get("target") != target
        or observation.get("procedure") != procedure
        or not isinstance(activity, Mapping)
        or activity.get("id") != job
        or activity.get("host") != f"edge-{target}"
        or activity.get("state") != "done"
        or activity.get("rc") != 0
        or start.get("job_id") != job
        or start.get("target") != target
        or start.get("procedure") != procedure
    ):
        raise ValueError("CAVIAR activity or observed start differs")
    scope = start.get("scope")
    if not isinstance(scope, str) or not scope or type(start.get("pid")) is not int or start["pid"] <= 0:
        raise ValueError("process identity differs")
    if (
        finish.get("scope") != scope
        or finish.get("runner_pid") != start["pid"]
        or finish.get("exit_code") != 0
        or finish.get("n_processes") != 1
        or finish.get("n_restarts") != 0
        or finish.get("journal_markers") != "matched"
        or not isinstance(start.get("container_digest"), str)
        or finish.get("container_digest") != start.get("container_digest")
    ):
        raise ValueError("finished process differs")
    verify_drain(drain, target=target, procedure=procedure, job_id=job, scope=scope)
    events = observation.get("events")
    if not isinstance(events, list):
        raise ValueError("CAVIAR events missing")
    started = [event for event in events if isinstance(event, Mapping) and event.get("type") == "job.started"]
    ended = [event for event in events if isinstance(event, Mapping) and event.get("type") == "job.ended"]
    if len(started) != 1 or len(ended) != 1:
        raise ValueError("CAVIAR start or terminal event missing")
    first, last = started[0], ended[0]
    if (
        type(first.get("seq")) is not int
        or type(last.get("seq")) is not int
        or first["seq"] >= last["seq"]
        or first.get("activity_id") != job
        or last.get("activity_id") != job
        or not isinstance(first.get("data"), Mapping)
        or first["data"].get("host") != f"edge-{target}"
        or not isinstance(first["data"].get("resources"), list)
        or f"device:edge-{target}" not in first["data"].get("resources", [])
        or not isinstance(last.get("data"), Mapping)
        or last["data"].get("rc") != 0
        or last["data"].get("state") != "done"
        or last["data"].get("detected_by") != "supervisor"
    ):
        raise ValueError("CAVIAR lease activity sequence differs")
    return first["seq"], last["seq"]


def native_causal_g2_binding(
    start, receipt, manifest, execution, candidate, journal, *, configuration_raw: bytes, target: str, procedure: str
) -> None:
    """Bind G3's observed execution to G2's candidate, configuration and ordered raw inputs."""
    from passage_eval.campaign.direct.raw_identity import task_from_full

    try:
        if not all(isinstance(item, Mapping) for item in (start, receipt, manifest, execution, candidate)):
            raise ValueError("G2 identity records are missing")
        config = pinned.strict_json_loads(configuration_raw, name="G2 configuration")
        target_candidate = candidate["targets"][target]
        pins = {
            "bundle": target_candidate["bundle_sha256"],
            "checkpoint": candidate["checkpoint_sha256"],
            "engine": target_candidate["engine_sha256"],
            "decoder": candidate["final_decoders_sha256"],
        }
        artifacts = {
            "checkpoint_sha256": pins["checkpoint"],
            "engine_sha256": pins["engine"],
            "final_decoders_sha256": pins["decoder"],
        }
        if (
            not all(pinned.is_hex64(digest) for digest in pins.values())
            or not isinstance(target_candidate.get("container_digest"), str)
            or not _CONTAINER_DIGEST.fullmatch(target_candidate["container_digest"])
            or execution != manifest.get("native_execution")
            or execution.get("artifacts") != artifacts
            or execution.get("container_digest") != target_candidate["container_digest"]
            or start.get("container_digest") != target_candidate["container_digest"]
            or receipt.get("native_artifact_hashes") != pins
            or receipt.get("journal_sha256") != pinned.sha256_bytes(journal)
            or not pinned.is_hex64(start.get("configuration_sha256"))
            or start.get("configuration_sha256") != pinned.sha256_bytes(configuration_raw)
            or start.get("configuration_sha256") != receipt.get("configuration_sha256")
        ):
            raise ValueError("G3 image, candidate, journal or configuration differs from G2")
        ids, specs = manifest["roster"], manifest["request_specs"]
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) != len(set(ids))
            or not isinstance(specs, Mapping)
            or set(ids) != set(specs)
        ):
            raise ValueError("G2 request manifest differs")
        if (
            not isinstance(config, Mapping)
            or config.get("target") != target
            or config.get("procedure") != procedure
            or config.get("seed") != manifest.get("seed")
            or config.get("dataset_split") != "validation"
            or config.get("artifact_pins") != pins
            or any(specs[sample_id].get("budget") != config.get("budget") for sample_id in ids)
        ):
            raise ValueError("G2 configuration differs from manifest, candidate or target")
        index = b"".join(
            pinned.canonical_json(task_from_full(sample_id, specs[sample_id])) + b"\n" for sample_id in ids
        )
        if (
            start.get("input_index_sha256") != receipt.get("input_index_sha256")
            or receipt.get("input_index_sha256") != config.get("index_sha256")
            or receipt.get("input_index_sha256") != pinned.sha256_bytes(index)
        ):
            raise ValueError("G3 input index differs from G2")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("G2 identity records are incomplete") from exc


class _Checker:
    def __init__(self, record: dict[str, Any], evidence_dir: Path, final_candidates: Path, now: _dt.datetime):
        self.record = record
        self.evidence_dir = evidence_dir
        self.final_candidates_path = final_candidates
        self.now = now
        self.decisions = record["protocol"]["decisions"]
        self.pins: dict[str, str] = {}
        self.external_pins: dict[str, str] = {}
        self._json_cache: dict[str, Any] = {}
        self.admitted: dict[str, Any] = {}
        self._source_roster: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------ evidence files
    def load(self, ref: Mapping[str, Any] | None, gate: _Gate, label: str, *, as_json: bool = True) -> Any:
        """The content of a referenced evidence file, or ``None`` after recording a reason."""
        if ref is None:
            gate.red(f"{label}_missing", "no evidence file is referenced")
            return None
        try:
            path, digest = pinned.file_ref(ref)
        except pinned.PinnedFileError as exc:
            gate.red(f"{label}_unverifiable", exc)
            return None
        if path in self._json_cache and self.pins.get(path) == digest:
            return copy.deepcopy(self._json_cache[path])
        try:
            location = pinned.resolve_inside(self.evidence_dir, path)
            raw = pinned.read_pinned(location, digest)
            content = pinned.strict_json_loads(raw, name=path) if as_json else raw
        except (pinned.PinnedFileError, OSError) as exc:
            gate.red(f"{label}_unverifiable", exc)
            return None
        self.pins[path] = digest
        self._json_cache[path] = content
        return copy.deepcopy(content)

    def decision(self, name: str) -> Any:
        return self.decisions[name]["value"]

    def sealed_decision(self, name: str) -> bool:
        entry = self.decisions[name]
        return entry["status"] == SEALED and not unsealed_markers(entry["value"])

    def native_persistent(self) -> bool:
        from passage_eval.execution import timing

        return (
            self.record["protocol"].get("profile") in NATIVE_PROFILES
            and self.decision("timing_profile") == timing.NATIVE_PERSISTENT
        )

    # ------------------------------------------------------------------ G1
    def g1(self) -> GateResult:
        gate = _Gate("G1")
        record = self.record
        profile = record["protocol"].get("profile", STRICT_PROFILE)
        self.admitted["profile"] = profile
        if profile in NATIVE_PROFILES:
            for problem in _descriptive_scope_problems(self.decisions, profile):
                gate.red("descriptive_scope_differs", problem)
        if profile == RULE8_NATIVE_PROFILE:
            from passage_eval.execution import timing

            if self.decision("timing_profile") != timing.NATIVE_PERSISTENT:
                gate.red("rule8_native_timing_differs", "this profile measures direct native raw arrays to path")
            analysis = self.decision("confirmatory_analysis")
            targets = self.decision("targets")
            if (
                not isinstance(analysis, Mapping)
                or analysis.get("seeds") != list(PLANNED_SEEDS)
                or not isinstance(targets, list)
                or analysis.get("target") not in targets
            ):
                gate.red(
                    "rule8_seed_analysis_missing",
                    "all three planned seeds and the target analysis must be sealed before the first pass",
                )
            if self.decision("strict_comparison") is not True:
                gate.red(
                    "rule8_strict_comparison_required", "the compared candidates must use identical recorded limits"
                )
        if record["status"] != SEALED:
            gate.red("record_not_sealed", f"record status is {record['status']!r}")
        document = self.load(record["protocol"]["document"], gate, "protocol_document", as_json=False)
        if document is not None:
            text = document.decode("utf-8", errors="replace")
            if "proposed_not_sealed" in text.lower():
                gate.red("protocol_document_has_unsealed_values", "the sealed protocol still says proposed_not_sealed")
        if unsealed_markers(record["protocol"]["version"]):
            gate.red("protocol_version_unsealed", record["protocol"]["version"])
        for name in DECISIONS:
            entry = self.decisions[name]
            if entry["status"] != SEALED:
                gate.red("decision_not_sealed", f"{name} ({entry['status']})")
                continue
            markers = unsealed_markers(entry["value"])
            if markers:
                gate.red("decision_has_unsealed_value", f"{name}: {markers}")
                continue
            problem = _decision_problem(name, entry["value"], self.decisions, profile=profile)
            if problem:
                gate.red("decision_invalid", f"{name}: {problem}")
        authorization = self.decisions["harness_commit_authorization"]["value"]
        if isinstance(authorization, Mapping) and authorization.get("commit") != record["harness"]["commit"]:
            gate.red("harness_commit_not_authorized", "the authorised commit differs from the record's harness commit")
        for reason in approval_problems(record, now=self.now):
            gate.red(reason["code"], reason["detail"])
        return gate.result()

    # ------------------------------------------------------------------ G2
    def g2(self) -> GateResult:
        gate = _Gate("G2")
        harness = self.record["harness"]
        if not pinned.is_hex40(harness["commit"]):
            gate.red("harness_commit_invalid", "a 40-hex commit is required")
        if not pinned.is_hex64(harness["source_sha256"]):
            gate.red("harness_source_hash_invalid", "a 64-hex source hash is required")
        commit, source = _running_harness()
        host_binding = None
        if (
            self.native_persistent()
            and self.record["protocol"].get("profile") == RULE8_NATIVE_PROFILE
            and commit is not None
            and commit != harness["commit"]
        ):
            host_binding = host_verifier_runtime_binding(harness, commit, verifier_source_sha256=source)
        if host_binding is not None:
            self.admitted["host_verifier_binding"] = {**host_binding, "verifier_source_sha256": source}
        if source != harness["source_sha256"] and host_binding is None:
            gate.red("running_harness_source_differs", "the running harness is not the pinned source")
        if commit is None:
            gate.red("running_harness_commit_unknown", "the running harness is not a git checkout")
        elif commit != harness["commit"] and host_binding is None:
            gate.red("running_harness_commit_differs", f"running {commit[:12]}")
        else:
            dirty = _uncommitted_harness_paths()
            if dirty is None:
                gate.red("running_harness_tree_unverifiable", "git status of the harness checkout failed")
            elif dirty:
                gate.red("running_harness_tree_dirty", f"{len(dirty)} uncommitted path(s), e.g. {dirty[:3]}")
        if self.native_persistent():
            return self._native_g2(gate)
        receipt = self.load(self.record["evidence"]["native_chain_rehearsal"], gate, "rehearsal_receipt")
        if receipt is None:
            return gate.result()
        if not isinstance(receipt, Mapping) or receipt.get("schema") != REHEARSAL_SCHEMA:
            gate.red("rehearsal_schema_invalid", f"{REHEARSAL_SCHEMA} required")
            return gate.result()
        markers = non_evidence_markers(receipt)
        if markers or _flags_unqualified(receipt):
            gate.red("rehearsal_evidence_not_real", f"markers {markers or ['qualified: false']}")
        if unsealed_markers(receipt):
            gate.red("rehearsal_has_unsealed_values", unsealed_markers(receipt))
        plan_sha256 = receipt.get("plan_sha256")
        if not pinned.is_hex64(plan_sha256):
            gate.red("rehearsal_plan_missing", "the receipt names no rehearsal plan_sha256")
            plan_sha256 = None
        if receipt.get("split") != "validation":
            gate.red("rehearsal_not_validation", receipt.get("split"))
        if receipt.get("target") != HOST:
            gate.red("rehearsal_not_host", receipt.get("target"))
        if receipt.get("harness") != {"commit": harness["commit"], "source_sha256": harness["source_sha256"]}:
            gate.red("rehearsal_harness_differs", "the rehearsal was not made with the pinned harness")
        for name in CANDIDATES:
            expected = self.record["candidates"][name]["checkpoint_sha256"]
            got = (receipt.get("candidates") or {}).get(name, {})
            if not isinstance(got, Mapping) or got.get("checkpoint_sha256") != expected:
                gate.red("rehearsal_candidate_differs", f"{name} checkpoint")
        if receipt.get("budget") != self.decision("budget"):
            gate.red("rehearsal_budget_differs", "the rehearsal budget is not the sealed budget")
        if receipt.get("timing_profile") != self.decision("timing_profile"):
            gate.red("rehearsal_timing_profile_differs", receipt.get("timing_profile"))
        warmup = receipt.get("warmup") or {}
        if not (
            isinstance(warmup, Mapping)
            and warmup.get("performed") is True
            and pinned.is_hex64(warmup.get("roster_sha256"))
        ):
            gate.red("rehearsal_warmup_not_performed", "no performed warm-up from a pinned roster")
        rewarm = receipt.get("rewarm") or {}
        if not (
            isinstance(rewarm, Mapping)
            and rewarm.get("exercised") is True
            and _is_int(rewarm.get("n_rewarm_events"))
            and rewarm["n_rewarm_events"] >= 1
        ):
            gate.red("rehearsal_rewarm_not_exercised", "no re-warm event was exercised")
        resume = receipt.get("resume") or {}
        if not (
            isinstance(resume, Mapping)
            and resume.get("exercised") is True
            and _is_int(resume.get("n_batches_resumed"))
            and resume["n_batches_resumed"] >= 1
            and resume.get("n_unplanned_relaunches") == 0
        ):
            gate.red("rehearsal_resume_not_exercised", "resume without unplanned relaunch was not exercised")
        # The R7 relaunch is a planned rule: the rehearsal must exercise it on purpose, on a deliberately killed
        # batch; it is counted apart from the unplanned relaunches above, which must be zero.
        kill = receipt.get("deliberate_kill") or {}
        if not (
            isinstance(kill, Mapping)
            and kill.get("exercised") is True
            and _is_int(kill.get("n_deliberate_kills"))
            and kill["n_deliberate_kills"] >= 1
            and _is_int(kill.get("n_planned_relaunches"))
            and kill["n_planned_relaunches"] >= kill["n_deliberate_kills"]
            and kill.get("killed_attempts_scored") == 0
        ):
            gate.red(
                "rehearsal_kill_not_exercised",
                "no deliberately killed batch was relaunched under the planned R7 rule with its killed attempt unscored",
            )
        procedures = receipt.get("procedures")
        sealed_order = self.decision("paired_order")
        expected_procedures = sorted(sealed_order.get("procedures", [])) if isinstance(sealed_order, Mapping) else None
        if not isinstance(procedures, Mapping) or not procedures:
            gate.red("rehearsal_procedures_missing", "no aggregated procedure")
            return gate.result()
        if expected_procedures is not None and sorted(procedures) != expected_procedures:
            gate.red("rehearsal_procedures_differ", f"{sorted(procedures)} != sealed {expected_procedures}")
        minimum = self.decision("min_warm_requests")
        minimum = minimum if _is_int(minimum) and minimum >= 1 else 100
        expected = {
            "harness": {"commit": harness["commit"], "source_sha256": harness["source_sha256"]},
            "budget": self.decision("budget"),
            "timing_profile": self.decision("timing_profile"),
            "seeds": self.decision("seeds"),
            "plan_sha256": plan_sha256,
            "min_warm_requests": minimum,
        }
        latency_ok = True
        references: dict[tuple[str, str], str] = {}
        campaigns: dict[str, Any] = {}
        for name, entry in sorted(procedures.items()):
            ref = entry.get("aggregate") if isinstance(entry, Mapping) else None
            if isinstance(ref, Mapping):
                for key in ("path", "sha256"):
                    if not isinstance(ref.get(key), str):
                        continue
                    other = references.setdefault((key, ref[key]), name)
                    if other != name:
                        gate.red("rehearsal_aggregate_shared", f"{name} and {other} name the same aggregate {key}")
            aggregate = self.load(ref, gate, f"aggregate_{name}")
            if aggregate is None:
                latency_ok = False
                continue
            problems = _aggregate_problems(aggregate, procedure=name, expected=expected)
            for code, detail in problems:
                gate.red(code, f"{name}: {detail}")
            latency_ok = latency_ok and not any(code in _LATENCY_UNSHOWN for code, _ in problems)
            if isinstance(aggregate, Mapping):
                campaigns[name] = {
                    key: aggregate.get(key)
                    for key in ("campaign_id", "admissible_roster_sha256", "reference_admission_chain_sha256")
                }
                campaigns[name]["population"] = aggregate.get("population")
        if len({pinned.canonical_sha256(value) for value in campaigns.values()}) > 1:
            gate.red(
                "rehearsal_aggregates_disagree",
                "the aggregates do not come from one rehearsal campaign (campaign, rosters, chain or population)",
            )
        self.admitted["rehearsal_latency_ok"] = latency_ok
        return gate.result()

    def _native_receipt(self, target: str, gate: _Gate) -> Mapping[str, Any] | None:
        receipt = self.load(self.record["evidence"]["edge"].get(target), gate, f"native_validation_{target}")
        if receipt is None:
            return None
        if not isinstance(receipt, Mapping) or receipt.get("schema") != NATIVE_VALIDATION_SCHEMA:
            gate.red("native_validation_schema_invalid", target)
            return None
        if non_evidence_markers(receipt) or _flags_unqualified(receipt):
            gate.red("native_validation_not_real", target)
        if (
            receipt.get("target") != target
            or receipt.get("split") != "validation"
            or receipt.get("harness") != self.record["harness"]
        ):
            gate.red("native_validation_identity_differs", target)
        if not pinned.is_hex64(receipt.get("plan_sha256")):
            gate.red("native_validation_plan_missing", target)
        if not isinstance(receipt.get("procedures"), Mapping) or set(receipt["procedures"]) != set(CANDIDATES):
            gate.red("native_validation_candidates_differ", target)
            return None
        return receipt

    def _native_g2(self, gate: _Gate) -> GateResult:
        """Real native validation replaces the supervised host rehearsal in the explicit native profiles."""
        from passage_eval.execution import timing

        targets = self.decision("targets")
        if not isinstance(targets, list) or not targets or any(target not in EDGE_TARGETS for target in targets):
            gate.red("native_validation_targets_invalid", "declared edge targets required")
            return gate.result()
        latency_ok = True
        warmup_projections = []
        for target in targets:
            receipt = self._native_receipt(target, gate)
            if receipt is None:
                latency_ok = False
                continue
            populations = []
            for name in CANDIDATES:
                entry = receipt["procedures"][name]
                if not isinstance(entry, Mapping):
                    gate.red("native_validation_candidate_invalid", f"{target}/{name}")
                    latency_ok = False
                    continue
                candidate = self.record["candidates"][name]
                if (
                    entry.get("completed") is not True
                    or entry.get("accelerator_used") is not True
                    or entry.get("checkpoint_sha256") != candidate["checkpoint_sha256"]
                    or entry.get("engine_sha256") != candidate["targets"].get(target, {}).get("engine_sha256")
                ):
                    gate.red(
                        "native_validation_candidate_differs",
                        f"{target}/{name}: native execution and pinned artifacts required",
                    )
                if self.record["protocol"].get("profile") == RULE8_NATIVE_PROFILE:
                    execution = entry.get("native_execution")
                    digest = candidate["targets"].get(target, {}).get("container_digest")
                    if (
                        not isinstance(execution, Mapping)
                        or not isinstance(digest, str)
                        or not _CONTAINER_DIGEST.fullmatch(digest)
                        or execution.get("container_digest") != digest
                    ):
                        gate.red(
                            "native_container_execution_differs",
                            f"{target}/{name}: emitted execution must name the sealed image digest",
                        )
                warmup = timing.manifest_warmup(
                    {
                        "series": "warm",
                        "timing_profile": timing.NATIVE_PERSISTENT,
                        "native_execution": entry.get("native_execution"),
                    }
                )
                if not warmup["performed"]:
                    gate.red(
                        "native_validation_warmup_invalid",
                        f"{target}/{name}: one process, one input, 16 initial warmups, no restart",
                    )
                aggregate = self.load(entry.get("aggregate"), gate, f"native_aggregate_{target}_{name}")
                if aggregate is None:
                    latency_ok = False
                    continue
                problems = _aggregate_problems(
                    aggregate,
                    procedure=name,
                    expected={
                        "harness": self.record["harness"],
                        "target": target,
                        "budget": self.decision("budget"),
                        "timing_profile": timing.NATIVE_PERSISTENT,
                        "seeds": self.decision("seeds"),
                        "plan_sha256": receipt.get("plan_sha256"),
                        "min_warm_requests": 1,
                    },
                )
                for code, detail in problems:
                    gate.red(code, f"{target}/{name}: {detail}")
                latency_ok = latency_ok and not any(code in _LATENCY_UNSHOWN for code, _ in problems)
                if isinstance(aggregate, Mapping):
                    projection = self._native_measurement_bindings(entry, aggregate, target, name, gate)
                    if projection is not None:
                        warmup_projections.append(projection)
                    sessions = aggregate.get("sessions")
                    if (
                        not isinstance(sessions, Mapping)
                        or not _is_int(sessions.get("n_sessions"))
                        or sessions["n_sessions"] != 1
                    ):
                        gate.red(
                            "native_validation_not_persistent",
                            f"{target}/{name}: fragments are not new process sessions",
                        )
                    populations.append(
                        {
                            key: aggregate.get(key)
                            for key in ("admissible_roster_sha256", "reference_admission_chain_sha256", "population")
                        }
                    )
            if len({pinned.canonical_sha256(population) for population in populations}) > 1:
                gate.red("native_validation_populations_differ", target)
        self.admitted["rehearsal_latency_ok"] = latency_ok
        self.admitted["native_warmup_projections"] = warmup_projections
        return gate.result()

    def _native_measurement_bindings(self, entry, aggregate, target, name, gate):
        from passage_eval.io import roster as roster_io
        from passage_eval.execution import timing

        measured = self.load(entry.get("roster"), gate, f"native_roster_{target}_{name}")
        warmup = self.load(entry.get("warmup_roster"), gate, f"native_warmup_roster_{target}_{name}")
        manifest = self.load(entry.get("run_manifest"), gate, f"native_run_manifest_{target}_{name}")
        if any(value is None for value in (measured, warmup, manifest)):
            return
        try:
            measured = roster_io.validate_roster(measured)
            warmup = roster_io.validate_roster(warmup)
            if (
                measured["dataset"]["split"] != "validation"
                or warmup["dataset"]["split"] != "validation"
                or len(warmup["tasks"]) != 1
            ):
                raise ValueError("native rehearsal and its one warm-up scene must use validation")
            self._roster_identity_problems(gate, "native_measured", measured)
            self._roster_identity_problems(gate, "native_warmup", warmup)
            a, b = native_input_projection(measured), native_input_projection(warmup)
            ids = {task["sample_id"] for task in a["tasks"]}
            contents = {task["content_identity_sha256"] for task in a["tasks"]}
            if ids.intersection(task["sample_id"] for task in b["tasks"]) or contents.intersection(
                task["content_identity_sha256"] for task in b["tasks"]
            ):
                raise ValueError("warm-up overlaps evaluated validation inputs by id or content")
            execution = entry["native_execution"]
            if timing.manifest_profile(manifest).name != timing.NATIVE_PERSISTENT:
                raise ValueError("emitted manifest does not declare native raw-array-to-path timing")
            if execution["warmup"]["input_roster_sha256"] != pinned.canonical_sha256(b):
                raise ValueError("warm-up projection differs from the verified validation roster")
            if execution.get("input_roster_sha256") != pinned.canonical_sha256(a):
                raise ValueError("measured input projection differs from the verified validation roster")
            if manifest.get("native_execution") != execution:
                raise ValueError("session identity or execution evidence differs from the emitted manifest")
            candidate = self.record["candidates"][name]
            artifacts = {
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "engine_sha256": candidate["targets"][target]["engine_sha256"],
                "final_decoders_sha256": candidate["final_decoders_sha256"],
            }
            if execution.get("artifacts") != artifacts:
                raise ValueError("emitted native artifact digests differ from the admitted candidate")
            if manifest.get("roster") != [task["sample_id"] for task in measured["tasks"]]:
                raise ValueError("emitted request order differs from the verified validation roster")
            if manifest.get("seed") != self.decision("seeds")[0]:
                raise ValueError("emitted training seed differs from its sealed single-seed record")
            specs = manifest.get("request_specs")
            if (
                not isinstance(specs, Mapping)
                or set(specs) != ids
                or any(
                    {key: specs[task["sample_id"]].get(key) for key in roster_io.IDENTITY_FIELDS}
                    != task["input_identity"]
                    or specs[task["sample_id"]].get("budget") != self.decision("budget")
                    for task in measured["tasks"]
                )
            ):
                raise ValueError("emitted request inputs or budgets differ from the verified validation roster")
            per_seed = aggregate["per_seed"][str(self.decision("seeds")[0])]
            if (
                aggregate.get("admissible_roster_sha256") != measured["roster_sha256"]
                or per_seed.get("native_execution") != execution
                or per_seed.get("run_manifest_sha256") != entry["run_manifest"]["sha256"]
                or aggregate["population"]["n_scored"] != len(measured["tasks"])
            ):
                raise ValueError("aggregate is not bound to this emitted session, artifacts and roster")
            hashes = manifest.get("artifact_hashes")
            if not isinstance(hashes, Mapping) or set(hashes) != {"requests.jsonl", "paths.npz"}:
                raise ValueError("emitted records and paths hashes are required")
            directory = Path(entry["run_manifest"]["path"]).parent
            for filename, digest in hashes.items():
                self.load(
                    {"path": str(directory / filename), "sha256": digest},
                    gate,
                    f"native_emitted_{target}_{name}_{filename}",
                    as_json=False,
                )
            return pinned.canonical_sha256(b)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            gate.red("native_measurement_binding_invalid", f"{target}/{name}: {exc}")

    def _native_g3(self) -> GateResult:
        """Bounded lease evidence and actual process/GPU release, without a hardware inventory or extra fault campaigns."""
        gate = _Gate("G3")
        targets = self.decision("targets")
        if not isinstance(targets, list) or not targets or any(target not in EDGE_TARGETS for target in targets):
            gate.red("targets_undeclared", "native edge targets required")
            return gate.result()
        self.admitted["targets"] = list(targets)
        order = self.decision("paired_order")
        order = order.get("procedures") if isinstance(order, Mapping) else None
        if not isinstance(order, list) or len(order) != 2 or set(map(str, order)) != set(CANDIDATES):
            gate.red("native_candidate_order_invalid", "the sealed Seg/Reg process order is required")
            return gate.result()
        for target in targets:
            receipt = self._native_receipt(target, gate)
            if receipt is None:
                continue
            previous_drain = None
            previous_causal_end = None
            causal_ledger = None
            previous_mode = None
            session_ids = set()
            for name in order:
                entry = receipt["procedures"][name]
                entry = entry if isinstance(entry, Mapping) else {}
                mode = "causal" if "cavops_observation" in entry else "timestamps"
                if previous_mode is not None and mode != previous_mode:
                    gate.red(
                        "native_lifecycle_mode_mixed",
                        f"{target}: candidate lifecycle evidence uses different orderings",
                    )
                    continue
                previous_mode = mode
                if mode == "causal":
                    from passage_eval.campaign.direct.oci_validation_relay import _journal_binding

                    observation = self.load(entry.get("cavops_observation"), gate, f"cavops_{target}_{name}")
                    start = self.load(entry.get("image_start"), gate, f"image_start_{target}_{name}")
                    finish = self.load(entry.get("image_finish"), gate, f"image_finish_{target}_{name}")
                    drain = self.load(entry.get("drain"), gate, f"drain_{target}_{name}")
                    journal = self.load(entry.get("journal"), gate, f"journal_{target}_{name}", as_json=False)
                    target_receipt = self.load(entry.get("target_receipt"), gate, f"target_receipt_{target}_{name}")
                    manifest = self.load(entry.get("run_manifest"), gate, f"run_manifest_{target}_{name}")
                    configuration = self.load(
                        entry.get("configuration"), gate, f"configuration_{target}_{name}", as_json=False
                    )
                    if any(
                        value is None
                        for value in (
                            observation,
                            start,
                            finish,
                            drain,
                            journal,
                            target_receipt,
                            manifest,
                            configuration,
                        )
                    ):
                        continue
                    try:
                        first, last = native_causal_lease(
                            observation, start, finish, drain, target=target, procedure=name
                        )
                        native_causal_g2_binding(
                            start,
                            target_receipt,
                            manifest,
                            entry.get("native_execution"),
                            self.record["candidates"][name],
                            journal,
                            configuration_raw=configuration,
                            target=target,
                            procedure=name,
                        )
                        ledger = observation.get("ledger_id")
                        if not isinstance(ledger, str) or not ledger:
                            raise ValueError("CAVIAR event ledger identity is missing")
                        if causal_ledger is not None and ledger != causal_ledger:
                            raise ValueError("candidate event ledgers differ")
                        markers = _journal_binding(journal, start.get("namespace_pid"))
                        if markers.get("journal_markers") != "matched" or markers.get(
                            "runner_session_uuid"
                        ) != finish.get("runner_session_uuid"):
                            raise ValueError("journal and observed process differ")
                        rows = [
                            pinned.strict_json_loads(line, name="journal event")
                            for line in journal.splitlines()
                            if line.strip()
                        ]
                        if not rows or rows[-1].get("kind") != "runtime_released":
                            raise ValueError("completed runtime release is missing")
                    except (ValueError, TypeError, KeyError) as exc:
                        gate.red("native_causal_evidence_invalid", f"{target}/{name}: {exc}")
                        continue
                    if previous_causal_end is not None and first <= previous_causal_end:
                        gate.red(
                            "native_candidates_overlap", f"{target}/{name}: prior job must end before the next starts"
                        )
                    previous_causal_end = last
                    causal_ledger = ledger
                    execution = entry.get("native_execution")
                    session_id = execution.get("session_id") if isinstance(execution, Mapping) else None
                    if (
                        not isinstance(session_id, str)
                        or session_id != pinned.sha256_bytes(journal)
                        or session_id in session_ids
                    ):
                        gate.red("native_process_session_reused", f"{target}/{name}")
                    if isinstance(session_id, str):
                        session_ids.add(session_id)
                    continue
                process, lease = entry.get("process"), entry.get("lease")
                if (
                    not isinstance(process, Mapping)
                    or process.get("process_exited") is not True
                    or process.get("gpu_context_released") is not True
                ):
                    gate.red("native_candidate_not_released", f"{target}/{name}")
                    continue
                if not isinstance(lease, Mapping) or not (
                    lease.get("resource") == f"device:edge-{target}"
                    and lease.get("holder_host") == f"edge-{target}"
                    and isinstance(lease.get("activity_id"), str)
                    and _ACTIVITY_ID.fullmatch(lease["activity_id"])
                    and isinstance(lease.get("job_id"), str)
                    and lease["job_id"].strip()
                ):
                    gate.red("native_lease_invalid", f"{target}/{name}")
                    continue
                try:
                    acquired, started, finished, drained, released = (
                        parse_timestamp(value)
                        for value in (
                            lease.get("acquired_at"),
                            process.get("started_at"),
                            process.get("finished_at"),
                            process.get("drained_at"),
                            lease.get("released_at"),
                        )
                    )
                except (ValueError, TypeError):
                    gate.red("native_lifecycle_timestamps_invalid", f"{target}/{name}")
                    continue
                if not acquired <= started <= finished <= drained <= released <= self.now:
                    gate.red("native_lifecycle_order_invalid", f"{target}/{name}")
                if previous_drain is not None and started < previous_drain:
                    gate.red(
                        "native_candidates_overlap", f"{target}/{name}: previous process/GPU must be released first"
                    )
                previous_drain = drained
                execution = entry.get("native_execution")
                session_id = execution.get("session_id") if isinstance(execution, Mapping) else None
                if not isinstance(session_id, str) or not session_id or session_id in session_ids:
                    gate.red("native_process_session_reused", f"{target}/{name}")
                if isinstance(session_id, str):
                    session_ids.add(session_id)
        return gate.result()

    # ------------------------------------------------------------------ G3
    def g3(self) -> GateResult:
        if self.native_persistent():
            return self._native_g3()
        gate = _Gate("G3")
        targets = self.decision("targets")
        if not self.sealed_decision("targets"):
            gate.red("targets_not_sealed", "the campaign targets are not sealed")
        if not isinstance(targets, list) or not targets or any(t not in TARGETS for t in targets):
            gate.red("targets_undeclared", "no valid target list")
            return gate.result()
        self.admitted["targets"] = list(targets)
        machine_ids: dict[str, str] = {}
        activities: dict[str, str] = {}
        for target in [t for t in EDGE_TARGETS if t in targets]:
            evidence = self.load(self.record["evidence"]["edge"][target], gate, f"edge_evidence_{target}")
            if evidence is None:
                continue
            for code, detail in self._edge_problems(target, evidence, gate):
                gate.red(code, f"{target}: {detail}")
            identity = evidence.get("target_identity") if isinstance(evidence, Mapping) else None
            lease = evidence.get("lease") if isinstance(evidence, Mapping) else None
            if isinstance(identity, Mapping) and isinstance(identity.get("machine_id_sha256"), str):
                other = machine_ids.setdefault(identity["machine_id_sha256"], target)
                if other != target:
                    gate.red("edge_target_identity_not_distinct", f"{target} and {other} share a machine id")
            if isinstance(lease, Mapping) and isinstance(lease.get("activity_id"), str):
                other = activities.setdefault(lease["activity_id"], target)
                if other != target:
                    gate.red("edge_lease_not_distinct", f"{target} and {other} share a lease activity")
        return gate.result()

    def _edge_problems(self, target: str, evidence: Any, gate: _Gate) -> list[tuple[str, str]]:
        problems: list[tuple[str, str]] = []
        if not isinstance(evidence, Mapping) or evidence.get("schema") != EDGE_EVIDENCE_SCHEMA:
            return [("edge_evidence_schema_invalid", f"{EDGE_EVIDENCE_SCHEMA} required")]
        markers = non_evidence_markers(evidence)
        if markers:
            problems.append(("edge_evidence_not_real", f"markers {markers}"))
        if _flags_unqualified(evidence):
            problems.append(("edge_evidence_not_real", "declares qualified: false or no target evidence"))
        if evidence.get("target") != target:
            problems.append(("edge_target_mismatch", evidence.get("target")))
        if evidence.get("harness") != dict(self.record["harness"]):
            problems.append(("edge_harness_differs", "the remote run did not use the pinned harness"))
        identity = evidence.get("target_identity")
        if not (
            isinstance(identity, Mapping)
            and isinstance(identity.get("hostname"), str)
            and identity["hostname"].strip()
            and pinned.is_hex64(identity.get("machine_id_sha256"))
            and pinned.is_hex64(identity.get("agent_sha256"))
            and isinstance(identity.get("ssh_host_key_sha256"), str)
            and identity["ssh_host_key_sha256"].startswith("SHA256:")
            and len(identity["ssh_host_key_sha256"]) > len("SHA256:") + 20
        ):
            problems.append(("edge_identity_attestation_invalid", "hostname, machine id, host key and agent digests"))
        lease = evidence.get("lease")
        run = evidence.get("run")
        drain = evidence.get("drain")
        times: dict[str, _dt.datetime] = {}
        if not (
            isinstance(lease, Mapping)
            and isinstance(lease.get("activity_id"), str)
            and _ACTIVITY_ID.fullmatch(lease["activity_id"])
            and isinstance(lease.get("job_id"), str)
            and lease["job_id"].strip()
            and isinstance(lease.get("resource"), str)
            and lease["resource"].strip()
            and lease.get("holder_host") == f"edge-{target}"
            and isinstance(lease.get("effect_key"), str)
            and lease["effect_key"].strip()
        ):
            problems.append(("edge_lease_invalid", f"activity job:<16 hex>, job id, resource, holder edge-{target}"))
            lease = None
        if not (isinstance(run, Mapping) and pinned.is_hex64(run.get("plan_sha256"))):
            problems.append(("edge_run_invalid", "run window and plan digest"))
            run = None
        try:
            if lease is not None:
                times["acquired"] = parse_timestamp(lease.get("acquired_at"))
                times["released"] = parse_timestamp(lease.get("released_at"))
            if run is not None:
                times["started"] = parse_timestamp(run.get("started_at"))
                times["finished"] = parse_timestamp(run.get("finished_at"))
            if isinstance(drain, Mapping):
                times["drained"] = parse_timestamp(drain.get("confirmed_at"))
        except (ValueError, TypeError) as exc:
            problems.append(("edge_timestamps_invalid", exc))
            return problems
        if not (
            isinstance(drain, Mapping)
            and drain.get("cgroup_empty") is True
            and drain.get("accelerator_idle") is True
            and drain.get("before_release") is True
        ):
            problems.append(("edge_drain_not_confirmed", "empty cgroup and idle accelerator before release"))
        if {"acquired", "released", "started", "finished", "drained"} <= times.keys():
            if not times["acquired"] <= times["started"] <= times["finished"] <= times["drained"] <= times["released"]:
                problems.append(("edge_lease_order_invalid", "acquire <= start <= finish <= drain <= release required"))
        log_ref = evidence.get("lease_verification_log")
        log_ref = log_ref if isinstance(log_ref, Mapping) else {}
        period = log_ref.get("period_s")
        log = self.load(
            {"path": log_ref.get("path"), "sha256": log_ref.get("sha256")} if log_ref.get("path") else None,
            gate,
            f"edge_lease_log_{target}",
            as_json=False,
        )
        if log is not None:
            problems.extend(self._lease_log_problems(target, log, period, log_ref.get("n_checks"), lease, times))
        for kind, outcome in (
            ("transport_loss_test", "fail_closed_lease_retained"),
            ("cancellation_test", "drained_before_release"),
        ):
            test = evidence.get(kind)
            receipt = self.load(
                test.get("receipt") if isinstance(test, Mapping) else None, gate, f"edge_{kind}_receipt_{target}"
            )
            if receipt is None:
                continue
            if non_evidence_markers(receipt) or _flags_unqualified(receipt):
                problems.append(("edge_evidence_not_real", f"{kind} receipt markers"))
            if not (
                isinstance(receipt, Mapping)
                and receipt.get("schema") == FAULT_TEST_SCHEMA
                and receipt.get("target") == target
                and receipt.get("kind") == kind
                and receipt.get("outcome") == outcome
                and test.get("outcome") == outcome
                and receipt.get("drain_confirmed") is True
                and receipt.get("lease_retained_until_drain") is True
                and isinstance(receipt.get("activity_id"), str)
                and _ACTIVITY_ID.fullmatch(receipt["activity_id"])
            ):
                problems.append((f"edge_{kind}_invalid", f"{FAULT_TEST_SCHEMA} with outcome {outcome}"))
        return problems

    @staticmethod
    def _lease_log_problems(target, raw: bytes, period, n_checks, lease, times) -> list[tuple[str, str]]:
        problems: list[tuple[str, str]] = []
        if not _is_number(period) or period <= 0:
            return [("edge_lease_log_invalid", "a positive verification period is required")]
        try:
            lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
            entries = [pinned.strict_json_loads(line.encode("utf-8"), name="lease log line") for line in lines]
        except (UnicodeDecodeError, pinned.PinnedFileError) as exc:
            return [("edge_lease_log_invalid", exc)]
        if not entries or n_checks != len(entries):
            return [("edge_lease_log_invalid", "the declared number of checks differs from the log")]
        if non_evidence_markers(entries):
            problems.append(("edge_evidence_not_real", f"lease log markers {non_evidence_markers(entries)}"))
        stamps = []
        for entry in entries:
            if not (
                isinstance(entry, Mapping)
                and entry.get("schema") == LEASE_CHECK_SCHEMA
                and entry.get("confirmed") is True
                and lease is not None
                and entry.get("activity_id") == lease["activity_id"]
                and entry.get("resource") == lease["resource"]
                and entry.get("holder_host") == f"edge-{target}"
            ):
                return problems + [("edge_lease_not_continuously_confirmed", "a check did not confirm this lease")]
            try:
                stamps.append(parse_timestamp(entry.get("checked_at")))
            except (ValueError, TypeError) as exc:
                return problems + [("edge_lease_log_invalid", exc)]
        if stamps != sorted(stamps):
            problems.append(("edge_lease_log_invalid", "checks are not in time order"))
        gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        if gaps and max(gaps) > LEASE_GAP_TOLERANCE * period:
            problems.append(("edge_lease_not_continuously_confirmed", f"largest gap {max(gaps):.1f} s"))
        if {"started", "drained"} <= times.keys() and not (
            stamps[0] <= times["started"] and stamps[-1] >= times["drained"]
        ):
            problems.append(("edge_lease_log_does_not_cover_run", "checks must cover the run and the drain"))
        return problems

    # ------------------------------------------------------------------ G4
    def g4(self) -> GateResult:
        gate = _Gate("G4")
        from passage_eval.io import roster as roster_io
        from passage_eval.protocol import reference_admission as ra

        refs = self.record["evidence"]["references"]
        loaded = {name: self.load(refs[name], gate, name) for name in _REFERENCE_KEYS}
        if any(value is None for value in loaded.values()):
            return gate.result()
        try:
            source = roster_io.validate_roster(loaded["source_roster"])
        except ValueError as exc:
            gate.red("source_roster_invalid", exc)
            return gate.result()
        self._source_roster = source
        descriptive = self.record["protocol"].get("profile") in DESCRIPTIVE_PROFILES
        if self.record["protocol"].get("profile") in NATIVE_PROFILES:
            historical = self.load(refs["historical_roster"], gate, "historical_roster")
            if historical is not None:
                try:
                    historical = roster_io.validate_roster(historical, expected_sha256=HISTORICAL_S_ROSTER_SHA256)
                except ValueError as exc:
                    gate.red("historical_roster_invalid", exc)
                else:
                    if historical["schema"] != roster_io.SCHEMA_V1:
                        gate.red("historical_roster_invalid", "the historical pin must identify a v1 roster")
                    # v1 and v2 have different artifact digests. Compare the ordered tasks' actual input identity.
                    keys = ("sample_id", "source_file", "row_index", "input_identity")

                    def tasks(roster):
                        return [{key: task[key] for key in keys} for task in roster["tasks"]]

                    inventories = {}
                    for label, roster in (("source", source), ("historical", historical)):
                        inventory = roster["selection"].get("population")
                        if (
                            not isinstance(inventory, Mapping)
                            or not _is_int(inventory.get("n_rows"))
                            or inventory["n_rows"] < len(roster["tasks"])
                            or not pinned.is_hex64(inventory.get("ids_sha256"))
                        ):
                            gate.red(
                                "roster_identifier_inventory_invalid",
                                f"{label}: full split identifier inventory required",
                            )
                            return gate.result()
                        inventories[label] = inventory
                    dataset_keys = ("repo", "commit", "config", "resolution", "split")
                    if (
                        any(historical["dataset"].get(key) != source["dataset"].get(key) for key in dataset_keys)
                        or historical["cell"] != source["cell"]
                        or historical["dataset_root_identity"] != source["dataset_root_identity"]
                        or historical["benchmark_id"] != source["benchmark_id"]
                        or any(
                            inventories["historical"][key] != inventories["source"][key]
                            for key in ("n_rows", "ids_sha256")
                        )
                        or tasks(historical) != tasks(source)
                    ):
                        gate.red(
                            "historical_roster_identity_differs",
                            "ordered ids, inputs, source rows, split inventory or dataset differ",
                        )
        dataset = source["dataset"]
        if dataset["split"] != self.record["split"]:
            gate.red("source_roster_split_differs", dataset["split"])
        if source["schema"] != roster_io.SCHEMA_V2:
            gate.red("source_roster_not_v2", source["schema"])
        selection = source["selection"]
        sealed_selection = self.decision("roster_selection")
        if not (
            isinstance(sealed_selection, Mapping)
            and {
                "rule": selection.get("key"),
                "salt": selection.get("salt"),
                "n": selection.get("n_selected"),
            }
            == dict(sealed_selection)
            and selection.get("n_expected") == selection.get("n_selected")
        ):
            gate.red(
                "source_roster_selection_differs",
                "the roster's label-free selection rule, salt or n differs from the sealed S selection",
            )
        sealed_population = self.decision("population")
        selected_ids = [task["sample_id"] for task in source["tasks"]]
        selected_digest = hashlib.sha256("".join(f"{sample_id}\n" for sample_id in selected_ids).encode()).hexdigest()
        if not (
            isinstance(sealed_population, Mapping)
            and len(selected_ids) == sealed_population.get("n_rows")
            and selected_digest == sealed_population.get("ids_sha256")
        ):
            gate.red(
                "source_roster_population_differs",
                "the selected roster S is not the sealed {n_rows, ids_sha256} population",
            )
        self._roster_identity_problems(gate, "source", source)
        table = roster_io.roster_block_table(source)
        if not table["complete"]:
            gate.red("source_roster_blocks_incomplete", f"{table['n_tasks_without_block']} task(s) without block")
        summary = source["geographic_blocks"] if isinstance(source["geographic_blocks"], Mapping) else {}
        if (
            source["schema"] == roster_io.SCHEMA_V2
            and summary.get("footprints_required") is not True
            and summary.get("unplaced_block") is None
        ):
            gate.red(
                "source_roster_footprints_not_required", "block analysis needs every footprint or the unplaced block"
            )
        sealed_blocks = self.decision("block_analysis")
        sealed_blocks = sealed_blocks if isinstance(sealed_blocks, Mapping) else {}
        if summary.get("rule", DEFAULT_BLOCK_RULE) != sealed_blocks.get("rule") or summary.get(
            "block_degrees"
        ) != sealed_blocks.get("block_degrees"):
            gate.red("source_roster_block_rule_differs", "the roster's block rule or degrees are not the sealed ones")
        unplaced_sealed = sealed_blocks.get("missing_footprint") == "unplaced_block"
        if unplaced_sealed != (summary.get("unplaced_block") is not None):
            gate.red(
                "source_roster_unplaced_block_differs",
                "the roster declares the unplaced block"
                if not unplaced_sealed
                else "the sealed unplaced block is absent",
            )
        try:
            policy = ra.validate_policy(loaded["policy"])
            chain = ra.admission_chain(source, policy, loaded["inventory"], loaded["admissible_roster"])
        except (ValueError, KeyError, TypeError) as exc:
            gate.red("admission_chain_invalid", exc)
            return gate.result()
        if chain != loaded["chain"]:
            gate.red("admission_chain_differs", "the recorded chain differs from the re-derived chain")
        if policy["reference_policy"] != self.decision("reference_policy"):
            gate.red("reference_policy_differs", policy["reference_policy"])
        if policy["reference_cohort_id"] != self.decision("reference_cohort_id"):
            gate.red("reference_cohort_differs", policy["reference_cohort_id"])
        if policy["reference_max_expansions"] != self.decision("reference_max_expansions"):
            gate.red("reference_limit_differs", policy["reference_max_expansions"])
        if pinned.canonical_sha256(policy["category_table"]) != self.decision("reference_category_table_sha256"):
            gate.red("category_table_differs", "the policy's table is not the sealed table")
        if unsealed_markers(policy["category_table_status"]):
            gate.red("category_table_not_sealed", policy["category_table_status"])
        caps = policy.get("exclusion_caps")
        sealed_caps = self.decision("reference_exclusion_caps")
        if not isinstance(caps, Mapping) or not isinstance(sealed_caps, Mapping):
            gate.red("exclusion_caps_missing", "the policy applies no exclusion caps")
        else:
            if {key: caps.get(key) for key in _EXCLUSION_CAP_KEYS} != {
                key: sealed_caps.get(key) for key in _EXCLUSION_CAP_KEYS
            }:
                gate.red("exclusion_caps_differ", "the policy's exclusion caps are not the sealed caps")
            if unsealed_markers(caps.get("status")):
                gate.red("exclusion_caps_not_sealed", caps.get("status"))
        admissible = roster_io.validate_roster(loaded["admissible_roster"])
        admissible_table = roster_io.roster_block_table(admissible)
        if not admissible_table["complete"] or admissible_table["function"] != source["geographic_blocks"].get(
            "function"
        ):
            gate.red("admissible_blocks_unsealed", "inherited blocks must be complete and bound to the block rule")
        cap = sealed_blocks.get("max_unplaced_fraction")
        n_unplaced = admissible_table.get("n_unplaced", 0)
        if not _is_number(cap):
            gate.red("block_analysis_invalid", "the sealed block_analysis has no numeric max_unplaced_fraction")
        self._roster_identity_problems(gate, "admissible", admissible)
        try:
            from passage_eval.protocol.block_census import adequacy

            block_adequacy = adequacy(
                admissible_table["blocks"],
                min_blocks=sealed_blocks["min_blocks"],
                max_block_share=sealed_blocks["max_block_share"],
                max_unplaced_fraction=sealed_blocks["max_unplaced_fraction"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            block_adequacy = {"meets_criterion": None, "error": str(exc)[:200]}
        # Judge R5: beyond the sealed cap, the block analysis of the population stops for diagnosis and no
        # confirmatory claim follows from it. The cap is reported, not gated: the run and its descriptive regret and
        # latency are not stopped (campaign.statistics then computes no block interval and no p-value).
        if not _is_number(cap):
            block_status = "invalid"
        elif n_unplaced > cap * admissible_table["n_tasks"]:
            if not descriptive:
                gate.red(
                    "unplaced_cap_exceeded",
                    f"{n_unplaced} unplaced task(s) exceed the sealed cap {cap:g} of {admissible_table['n_tasks']} tasks",
                )
            block_status = "halted_unplaced_cap"
        elif block_adequacy.get("meets_criterion") is True:
            block_status = "meets_criterion"
        else:
            block_status = "descriptive_only"
        self.admitted.update(
            source_roster_sha256=source["roster_sha256"],
            admissible_roster_sha256=admissible["roster_sha256"],
            admissible_roster=admissible,
            admissible_dataset_root_sha256=admissible["dataset_root_identity"]["sha256"],
            chain_sha256=chain["chain_sha256"],
            block_adequacy=block_adequacy,
            block_analysis={
                "status": block_status,
                "n_unplaced": n_unplaced,
                "n_tasks": admissible_table["n_tasks"],
                "max_unplaced_fraction": cap,
                "confirmatory_block_analysis_possible": not descriptive and block_status == "meets_criterion",
                "rule": "beyond the sealed unplaced cap the block analysis stops for diagnosis (judge R5); "
                "the run and its descriptive results do not",
            },
            reference_policy={
                key: policy[key] for key in ("reference_policy", "reference_cohort_id", "reference_max_expansions")
            },
            reference_ids={
                task["sample_id"]: task["reference_id"]
                for task in loaded["inventory"]["tasks"]
                if task.get("category") == "admissible"
            },
        )
        return gate.result()

    def _roster_identity_problems(self, gate: _Gate, label: str, roster: Mapping[str, Any]) -> None:
        """A roster of the sealed dataset revision, dataset root and evaluation cell (G4, and G5 for the warm-up)."""
        revision = self.decision("dataset_revision")
        keys = ("repo", "commit", "config", "resolution")
        if not (
            isinstance(revision, Mapping)
            and all(roster["dataset"].get(key) == revision.get(key) for key in keys)
            and all(key in revision for key in keys)
        ):
            gate.red(f"{label}_roster_revision_differs", "dataset repo, commit, config or resolution")
        validation_input = self.native_persistent() and label in ("native_measured", "native_warmup", "warmup")
        validation_root = self.decision("warmup") if validation_input else None
        expected_root = (
            (validation_root.get("dataset_root_sha256") if isinstance(validation_root, Mapping) else None)
            if validation_input
            else self.decision("dataset_root_sha256")
        )
        if roster["dataset_root_identity"].get("sha256") != expected_root:
            gate.red(f"{label}_roster_dataset_root_differs", "the roster pins another dataset root")
        cell = self.decision("evaluation_cell")
        if not (
            isinstance(cell, Mapping)
            and all(key in cell and roster["cell"].get(key) == cell[key] for key in _CELL_KEYS)
        ):
            gate.red(f"{label}_roster_cell_differs", "cost model, cost weight or task variant")

    # ------------------------------------------------------------------ G5
    def g5(self) -> GateResult:
        gate = _Gate("G5")
        from passage_eval.io import roster as roster_io
        from passage_eval.metrics import latency
        from passage_eval.protocol import warmup as warmup_io

        rule = self.decision("censoring_rule")
        if not (isinstance(rule, Mapping) and rule.get("id") == latency.RULE_ID):
            gate.red("censoring_rule_not_installed", f"the sealed rule must be the installed {latency.RULE_ID}")
        if not self.sealed_decision("censoring_rule"):
            gate.red("censoring_rule_not_sealed", "the censoring rule is not sealed")
        if not self.sealed_decision("min_warm_requests"):
            gate.red("min_warm_requests_not_sealed", "the minimum of warm requests is not sealed")
        if self.admitted.get("rehearsal_latency_ok") is not True:
            gate.red("rehearsal_latency_not_shown", "no rehearsal latency block meets the sealed minimum (see G2)")
        refs = self.record["evidence"]["warmup"]
        roster = self.load(refs["roster"], gate, "warmup_roster")
        selection = self.load(refs["selection"], gate, "warmup_selection")
        if roster is None or selection is None:
            return gate.result()
        try:
            roster = roster_io.validate_roster(roster)
        except ValueError as exc:
            gate.red("warmup_roster_invalid", exc)
            return gate.result()
        sealed = self.decision("warmup") if isinstance(self.decision("warmup"), Mapping) else {}
        if roster["dataset"]["split"] != warmup_io.WARMUP_SPLIT or roster["schema"] != roster_io.SCHEMA_V2:
            gate.red("warmup_roster_not_validation_v2", f"{roster['schema']} {roster['dataset']['split']}")
        if roster["selection"].get("salt") != sealed.get("salt"):
            gate.red("warmup_salt_differs", "the warm-up roster's salt is not the sealed salt")
        if len(roster["tasks"]) != sealed.get("k"):
            gate.red("warmup_size_differs", f"{len(roster['tasks'])} != sealed k")
        if self.native_persistent():
            digest = pinned.canonical_sha256(native_input_projection(roster))
            projections = self.admitted.get("native_warmup_projections")
            if not projections or any(projection != digest for projection in projections):
                gate.red(
                    "native_warmup_projection_differs", "G5 warm-up inputs differ from the verified native rehearsal"
                )
        if roster["dataset_root_identity"]["sha256"] != sealed.get("dataset_root_sha256"):
            gate.red("warmup_dataset_root_differs", "the warm-up roster pins another dataset root")
        admissible_root = self.admitted.get("admissible_dataset_root_sha256")
        if (
            not self.native_persistent()
            and admissible_root is not None
            and roster["dataset_root_identity"]["sha256"] != admissible_root
        ):
            gate.red("warmup_dataset_root_not_admissible_root", "the admissible roster of G4 pins another root")
        self._roster_identity_problems(gate, "warmup", roster)
        derivation = roster["selection"].get("warmup") or {}
        admissible = self.admitted.get("admissible_roster_sha256")
        if self.native_persistent():
            measured = self.admitted.get("admissible_roster")
            if (
                measured is None
                or self._source_roster is None
                or not native_warmup_disjoint(roster, measured)
                or not native_warmup_disjoint(roster, self._source_roster)
            ):
                gate.red(
                    "warmup_not_disjoint_from_admissible_roster",
                    "warm-up ID or input content overlaps S or the admissible roster, or a roster is unavailable",
                )
        elif admissible is None or derivation.get("evaluated_roster_sha256") != admissible:
            gate.red("warmup_not_disjoint_from_admissible_roster", "built against another evaluated roster")
        if derivation.get("disjointness_checked_by") != warmup_io.CONTENT_CHECK:
            gate.red("warmup_disjointness_by_content_missing", derivation.get("disjointness_checked_by"))
        if not isinstance(selection, Mapping) or selection.get("schema") != warmup_io.SELECTION_SCHEMA:
            gate.red("warmup_selection_invalid", f"{warmup_io.SELECTION_SCHEMA} required")
        else:
            digest = pinned.canonical_sha256({k: v for k, v in selection.items() if k != "selection_sha256"})
            if digest != selection.get("selection_sha256") or digest != derivation.get("selection_sha256"):
                gate.red("warmup_selection_differs", "selection digest differs from the roster's")
            if selection.get("selected_ids") != [task["sample_id"] for task in roster["tasks"]]:
                gate.red("warmup_selection_differs", "selected ids differ from the warm-up roster")
        return gate.result()

    def _container_reconstruction_problems(self, reconstruction: Any, name: str, target: str) -> list[str]:
        """Pin a rebuild description and its real files; the gate cannot authenticate publication or a build."""
        gate = _Gate("G6")
        if not isinstance(reconstruction, Mapping):
            return ["a pinned container reconstruction description is required"]
        entry = self.record["candidates"][name]["targets"][target]
        expected = {
            "schema": "passage-container-reconstruction/v1",
            "target": target,
            "container_digest": entry["container_digest"],
            "harness": self.record["harness"],
        }
        problems = [f"{key} differs" for key, value in expected.items() if reconstruction.get(key) != value]
        if reconstruction_non_evidence_markers(reconstruction) or _flags_unqualified(reconstruction):
            problems.append("the reconstruction description is marked as non-evidence")
        repository = reconstruction.get("source_repository")
        if not isinstance(repository, str) or not repository.startswith(("https://", "ssh://", "git@")):
            problems.append("the published source repository must be named")
        command = reconstruction.get("build_command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            problems.append("the exact build command is required")
        self.load(reconstruction.get("recipe"), gate, f"container_recipe_{name}_{target}", as_json=False)
        dependencies = reconstruction.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            problems.append("pinned runtime dependency or base-image lock files are required")
        else:
            for index, ref in enumerate(dependencies):
                self.load(ref, gate, f"container_dependency_{name}_{target}_{index}", as_json=False)
        problems.extend(f"{reason['code']}: {reason['detail']}" for reason in gate.reasons)
        return problems

    # ------------------------------------------------------------------ G6
    def g6(self) -> GateResult:
        gate = _Gate("G6")
        descriptive = self.record["protocol"].get("profile") in DESCRIPTIVE_PROFILES
        frozen = None
        try:
            raw = pinned.read_bounded(self.final_candidates_path)
            frozen = pinned.strict_json_loads(raw, name=self.final_candidates_path.name)
        except (pinned.PinnedFileError, OSError) as exc:
            gate.red("final_candidates_unavailable", exc)
        else:
            self.external_pins[str(self.final_candidates_path.resolve())] = pinned.sha256_bytes(raw)
            committed = _committed_file(self.record["harness"]["commit"], FINAL_CANDIDATES_RELATIVE)
            if committed is None:
                gate.red("final_candidates_unverifiable", "the file cannot be read at the record's harness commit")
            elif committed != raw:
                gate.red("final_candidates_not_committed", "the file differs from the one at the record's commit")
        installed = _installed_final_decoders()
        installed_sha = pinned.sha256_bytes(installed.read_bytes()) if installed.is_file() else None
        if installed_sha is not None:
            self.external_pins[str(installed)] = installed_sha
        if installed_sha != FROZEN_FINAL_DECODERS_SHA256:
            gate.red("final_decoders_changed", "the installed final_decoders.py differs from the frozen SHA-256")
        if isinstance(frozen, Mapping):
            frozen_sha = (frozen.get("reference_decoders") or {}).get("module_sha256")
            if frozen_sha != FROZEN_FINAL_DECODERS_SHA256:
                gate.red("final_candidates_decoder_pin_differs", "final-candidates.json pins another final_decoders.py")
        targets = self.decision("targets") if isinstance(self.decision("targets"), list) else []
        if not targets:
            gate.red("targets_undeclared", "bundle, engine and container digests cannot be checked")
        trees: dict[str, str] = {}
        catalog = None
        seeds = self.decision("seeds")
        replica_seed = seeds[0] if isinstance(seeds, list) and len(seeds) == 1 else None
        profile = self.record["protocol"].get("profile")
        if profile == RULE8_NATIVE_PROFILE or (profile in REPLICA_PROFILES and replica_seed != 2026):
            from passage_eval.protocol.replicas import ReplicaCatalogError, validate_replica_catalog

            value = self.load(self.record["evidence"].get("replica_catalog"), gate, "replica_catalog")
            if value is not None and isinstance(frozen, Mapping):
                try:
                    catalog = validate_replica_catalog(
                        value, final_candidates=frozen, final_candidates_sha256=pinned.sha256_bytes(raw)
                    )
                    if profile == RULE8_NATIVE_PROFILE and any(
                        entry["status"] != SEALED for entry in catalog["replicas"].values()
                    ):
                        gate.red(
                            "replica_catalog_incomplete",
                            "this approved S campaign pins all three seeds before the first pass",
                        )
                except (ReplicaCatalogError, KeyError, TypeError) as exc:
                    gate.red("replica_catalog_invalid", exc)
        for name in CANDIDATES:
            candidate = self.record["candidates"][name]
            if isinstance(frozen, Mapping):
                family = (frozen.get("candidates") or {}).get(name) or {}
                checkpoint = family.get("checkpoint_256") or {}
                if self.record["protocol"].get("profile") in REPLICA_PROFILES:
                    seeds = self.decision("seeds")
                    seed = seeds[0] if isinstance(seeds, list) and len(seeds) == 1 else None
                    if seed != checkpoint.get("seed"):
                        from passage_eval.protocol.replicas import ReplicaCatalogError, replica_identity

                        try:
                            identity = replica_identity(catalog, name, seed) if catalog is not None else None
                            checkpoint = (
                                {"seed": seed, "sha256": identity["checkpoint_sha256"]} if identity is not None else {}
                            )
                        except (ReplicaCatalogError, KeyError, TypeError):
                            checkpoint = {}
                    if checkpoint.get("seed") != seed or not pinned.is_hex64(checkpoint.get("sha256")):
                        gate.red(
                            "seed_checkpoint_not_frozen",
                            f"{name}: seed {seed} has no separately pinned checkpoint identity",
                        )
                    if seed != 2026 and checkpoint.get("sha256") == (family.get("checkpoint_256") or {}).get("sha256"):
                        gate.red(
                            "seed_checkpoint_relabelled", f"{name}: the 2026 checkpoint cannot stand in for seed {seed}"
                        )
                expected = checkpoint.get("sha256")
                if candidate["checkpoint_sha256"] != expected:
                    gate.red("checkpoint_differs_from_final_candidates", name)
            if candidate["final_decoders_sha256"] != installed_sha or installed_sha is None:
                gate.red("final_decoders_pin_differs", name)
            if candidate["decoder"] not in FROZEN_DECODER_ENTRIES[name]:
                gate.red("decoder_entry_not_frozen", f"{name}: {candidate['decoder']!r}")
            elif not pinned.is_hex64(candidate["decoder_tree_sha256"]):
                gate.red("decoder_tree_pin_missing", name)
            else:
                try:
                    tree = trees.get(candidate["decoder"])
                    if tree is None:
                        binding = self.admitted.get("host_verifier_binding")
                        if binding and candidate["decoder"].split(".", 1)[0] == "passage_eval":
                            tree = committed_decoder_tree_sha256(binding["runtime_commit"])
                        else:
                            tree = decoder_tree_sha256(candidate["decoder"])
                    trees[candidate["decoder"]] = tree
                except (ValueError, OSError) as exc:
                    gate.red("decoder_tree_unverifiable", f"{name}: {exc}")
                else:
                    if tree != candidate["decoder_tree_sha256"]:
                        gate.red("decoder_tree_differs", f"{name}: installed decoder tree is not the pinned tree")
            if not isinstance(candidate["predictor"], str) or ":" not in candidate["predictor"]:
                gate.red("predictor_entry_invalid", name)
            for target in targets:
                entry = candidate["targets"].get(target)
                if entry is None:
                    gate.red("target_artifacts_missing", f"{name} on {target}")
                    continue
                if not pinned.is_hex64(entry["bundle_sha256"]):
                    gate.red("bundle_digest_missing", f"{name} on {target}")
                if target in EDGE_TARGETS:
                    if not pinned.is_hex64(entry["engine_sha256"]):
                        gate.red("engine_digest_missing", f"{name} on {target}")
                    if not descriptive and not (
                        isinstance(entry["container_digest"], str)
                        and _CONTAINER_DIGEST.fullmatch(entry["container_digest"])
                    ):
                        gate.red("container_digest_missing", f"{name} on {target}")
                    if descriptive:
                        if entry["container_digest"] is not None:
                            gate.red(
                                "native_container_identity_not_applicable",
                                f"{name} on {target}: native execution declares no container",
                            )
                    if descriptive or (
                        self.record["protocol"].get("profile") == RULE8_NATIVE_PROFILE and "native_environment" in entry
                    ):
                        environment = self.load(
                            entry["native_environment"], gate, f"native_environment_{name}_{target}"
                        )
                        if environment is not None:
                            for problem in _native_environment_problems(
                                environment, self.record, name, target, self.now
                            ):
                                gate.red("native_environment_invalid", f"{name} on {target}: {problem}")
                    if self.record["protocol"].get("profile") == RULE8_NATIVE_PROFILE:
                        reconstruction = self.load(
                            entry["container_reconstruction"], gate, f"container_reconstruction_{name}_{target}"
                        )
                        if reconstruction is not None:
                            for problem in self._container_reconstruction_problems(reconstruction, name, target):
                                gate.red("container_reconstruction_invalid", f"{name} on {target}: {problem}")
                if entry["bundle"] is None:
                    gate.red("bundle_file_missing", f"{name} on {target}: the bundle file binds the checkpoint")
                else:
                    bundle = self.load(entry["bundle"], gate, f"bundle_{name}_{target}")
                    if bundle is not None:
                        if pinned.file_ref(entry["bundle"])[1] != entry["bundle_sha256"]:
                            gate.red("bundle_digest_differs", f"{name} on {target}")
                        lineage = bundle.get("lineage") if isinstance(bundle, Mapping) else None
                        if (
                            not isinstance(lineage, Mapping)
                            or lineage.get("checkpoint_sha256") != candidate["checkpoint_sha256"]
                        ):
                            gate.red("bundle_checkpoint_differs", f"{name} on {target}")
        return gate.result()


def _decision_problem(
    name: str, value: Any, decisions: Mapping[str, Any], *, profile: str = STRICT_PROFILE
) -> str | None:
    """Why a sealed decision value is invalid, or ``None``."""
    from passage_eval.execution import timing
    from passage_eval.protocol.reference_admission import REFERENCE_POLICIES

    native = (
        profile in NATIVE_PROFILES and (decisions.get("timing_profile") or {}).get("value") == timing.NATIVE_PERSISTENT
    )

    if (
        profile == RULE8_NATIVE_PROFILE
        and native
        and name in {"deadline_grace_ms", "hard_timeout_ms", "startup_timeout_s", "worker_rlimits", "batching"}
        and value == {"not_applicable": "native_direct_persistent"}
    ):
        return None

    if name == "dataset_revision":
        if not isinstance(value, Mapping) or set(value) != {"repo", "commit", "config", "resolution"}:
            return "needs exactly repo, commit, config, resolution"
        if not isinstance(value["repo"], str) or not value["repo"].strip():
            return "repo must be named"
        if not pinned.is_hex40(value["commit"]):
            return "commit must be a 40-hex dataset commit"
        if not _is_int(value["resolution"]) or value["resolution"] < 1:
            return "resolution must be a positive integer"
        return None if value["config"] == f"{value['resolution']}x{value['resolution']}" else "config must be <N>x<N>"
    if name == "dataset_root_sha256":
        return None if pinned.is_hex64(value) else "must be a SHA-256"
    if name == "evaluation_cell":
        if not isinstance(value, Mapping) or set(value) != set(_CELL_KEYS):
            return f"needs exactly {list(_CELL_KEYS)}"
        if not isinstance(value["cost_model"], str) or not value["cost_model"].strip():
            return "cost_model must be named"
        if not _is_number(value["cost_weight"]):
            return "cost_weight must be a finite number"
        return None if value["task_variant"] in ("obstacles", "free") else "task_variant must be obstacles or free"
    if name == "roster_selection":
        if not isinstance(value, Mapping) or set(value) != set(_ROSTER_SELECTION_KEYS):
            return f"needs exactly {list(_ROSTER_SELECTION_KEYS)}"
        if value["rule"] != ROSTER_SELECTION_RULE:
            return "rule must be the sealed SHA-256 roster ordering"
        if not isinstance(value["salt"], str) or not value["salt"]:
            return "salt is required"
        return None if _is_int(value["n"]) and value["n"] >= 1 else "n must be a positive integer"
    if name == "population":
        if not isinstance(value, Mapping) or set(value) != {"n_rows", "ids_sha256"}:
            return "needs exactly n_rows, ids_sha256"
        if not _is_int(value["n_rows"]) or value["n_rows"] < 1:
            return "n_rows must be a positive integer"
        return None if pinned.is_hex64(value["ids_sha256"]) else "ids_sha256 must be a SHA-256"
    if name == "budget":
        if not isinstance(value, Mapping) or set(value) != {
            "wall_time_ms",
            "max_expansions",
            "max_queue_operations",
            "max_live_states",
        }:
            return "needs exactly wall_time_ms, max_expansions, max_queue_operations, max_live_states"
        if not _is_number(value["wall_time_ms"]) or value["wall_time_ms"] <= 0:
            return "wall_time_ms must be positive"
        for key in ("max_expansions", "max_queue_operations", "max_live_states"):
            if value[key] is not None and (not _is_int(value[key]) or value[key] < 1):
                return f"{key} must be a positive integer or null"
        return None
    if name == "timing_profile":
        allowed = (
            (timing.SERVICE, timing.SERVICE_REWARM, timing.NATIVE_PERSISTENT)
            if profile in NATIVE_PROFILES
            else (timing.SERVICE, timing.SERVICE_REWARM)
        )
        return None if value in allowed else "a declared service profile or descriptive native profile is required"
    if name == "timing_boundary_id":
        profile = decisions["timing_profile"]["value"]
        if profile not in timing.PROFILES:
            return "the timing profile is unknown"
        return None if value == timing.get_profile(profile).boundary else "differs from the profile's boundary"
    if name in ("deadline_grace_ms", "hard_timeout_ms", "startup_timeout_s"):
        return None if _is_number(value) and value > 0 else "must be a positive number"
    if name == "max_restarts":
        if not _is_int(value) or value < 0:
            return "must be a nonnegative integer"
        batching = (decisions.get("batching") or {}).get("value")
        if (
            not native
            and isinstance(batching, Mapping)
            and _is_int(batching.get("batch_size"))
            and value < batching["batch_size"]
        ):
            return "must be at least the sealed batch size (R7: the restart cap never decides a task status)"
        return None
    if name == "worker_rlimits":
        if not isinstance(value, Mapping) or set(value) != {"nofile", "fsize_mb", "as_mb", "cpu_s"}:
            return "needs exactly nofile, fsize_mb, as_mb, cpu_s"
        bad = [k for k, v in value.items() if v is not None and (not _is_int(v) or v < 1)]
        return f"{bad} must be positive integers or null" if bad else None
    if name == "reference_policy":
        return None if value in REFERENCE_POLICIES else f"must be one of {list(REFERENCE_POLICIES)}"
    if name == "reference_cohort_id":
        policy = decisions["reference_policy"]["value"]
        if policy == "released_labels":
            return None if value is None else "released_labels uses no cohort (null)"
        return None if isinstance(value, str) and value.strip() else "the independent policy needs a named cohort"
    if name == "reference_max_expansions":
        return None if _is_int(value) and value >= 1 else "must be a positive integer"
    if name == "reference_category_table_sha256":
        return None if pinned.is_hex64(value) else "must be a SHA-256"
    if name == "reference_exclusion_caps":
        if not isinstance(value, Mapping) or set(value) != set(_EXCLUSION_CAP_KEYS):
            return f"needs exactly {list(_EXCLUSION_CAP_KEYS)}"
        if not all(_is_number(value[key]) and 0 <= value[key] < 1 for key in _EXCLUSION_CAP_KEYS[:2]):
            return "the fractions must lie in [0, 1)"
        return (
            None
            if _is_int(value["min_block_exclusions"]) and value["min_block_exclusions"] >= 1
            else ("min_block_exclusions must be a positive integer")
        )
    if name == "block_analysis":
        if not isinstance(value, Mapping) or set(value) != _BLOCK_ANALYSIS_KEYS:
            return f"needs exactly {sorted(_BLOCK_ANALYSIS_KEYS)}"
        if value["rule"] not in BLOCK_RULES:
            return f"rule must be one of {list(BLOCK_RULES)}"
        degrees = value["block_degrees"]
        if not _is_int(degrees) or degrees < 1 or 180 % degrees or 360 % degrees:
            return "block_degrees must divide 180 and 360"
        if value["missing_footprint"] not in ("halt", "unplaced_block"):
            return "missing_footprint must be halt or unplaced_block"
        if not _is_int(value["min_blocks"]) or value["min_blocks"] < 2:
            return "min_blocks must be an integer of at least 2"
        if not _is_number(value["max_block_share"]) or not 0 < value["max_block_share"] <= 1:
            return "max_block_share must lie in (0, 1]"
        if not _is_number(value["max_unplaced_fraction"]) or not 0 <= value["max_unplaced_fraction"] < 1:
            return "max_unplaced_fraction must lie in [0, 1)"
        return None
    if name == "censoring_rule":
        return None if isinstance(value, Mapping) and isinstance(value.get("id"), str) else "needs {id}"
    if name == "min_warm_requests":
        return None if _is_int(value) and value >= 1 else "must be a positive integer"
    if name in _POLICY_TEXT_DECISIONS:
        return None if isinstance(value, str) and value.strip() else "must be a stated policy"
    if name == "paired_order":
        if not isinstance(value, Mapping) or set(value) != {"granularity", "procedures", "schedule_seed"}:
            return "needs exactly granularity, procedures, schedule_seed"
        procedures = value["procedures"]
        if native and value["granularity"] == "per_candidate":
            pass
        elif value["granularity"] not in ("per_batch", "per_scene"):
            return "granularity must be per_batch or per_scene"
        if not isinstance(procedures, list) or not procedures or len(set(map(str, procedures))) != len(procedures):
            return "procedures must be unique names"
        seed = value["schedule_seed"]
        return None if _is_int(seed) or (isinstance(seed, str) and seed.strip()) else "schedule_seed is required"
    if name == "batching":
        from passage_eval.protocol.batching import MIN_SAFETY_MARGIN

        keys = {
            "batch_size",
            "lease_or_timeout_ms",
            "fixed_overhead_ms",
            "per_seed_overhead_ms",
            "per_request_overhead_ms",
            "persistence_ms",
            "observed_request_ms",
            "safety_margin",
        }
        if not isinstance(value, Mapping) or set(value) != keys:
            return f"needs exactly {sorted(keys)}"
        if not _is_int(value["batch_size"]) or not 1 <= value["batch_size"] <= MAX_BATCH_SIZE:
            return f"batch_size must be 1..{MAX_BATCH_SIZE}"
        if not _is_number(value["lease_or_timeout_ms"]) or value["lease_or_timeout_ms"] <= 0:
            return "lease_or_timeout_ms must be positive"
        if not _is_number(value["safety_margin"]) or value["safety_margin"] < MIN_SAFETY_MARGIN:
            return f"safety_margin must be at least {MIN_SAFETY_MARGIN:g} (R7)"
        observed = value["observed_request_ms"]
        if not (
            isinstance(observed, Mapping)
            and observed
            and all(isinstance(k, str) and _is_number(v) and v > 0 for k, v in observed.items())
        ):
            return "observed_request_ms must map every procedure to a positive measured time"
        numbers = keys - {"batch_size", "lease_or_timeout_ms", "observed_request_ms", "safety_margin"}
        bad = [k for k in numbers if not _is_number(value[k]) or value[k] < 0]
        return f"{sorted(bad)} must be nonnegative" if bad else None
    if name == "r7_policy":
        if native:
            return (
                None
                if value == NATIVE_INCOMPLETE_POLICY
                else "native sessions abort on interruption without automatic replay"
            )
        keys = {
            "max_ambiguous_attempts",
            "single_task_candidate_failures",
            "relaunch_stop_fraction",
            "candidate_attributable",
        }
        if not isinstance(value, Mapping) or set(value) != keys:
            return f"needs exactly {sorted(keys)}"
        if not _is_int(value["max_ambiguous_attempts"]) or value["max_ambiguous_attempts"] < 1:
            return "max_ambiguous_attempts must be a positive integer"
        if not _is_int(value["single_task_candidate_failures"]) or value["single_task_candidate_failures"] < 1:
            return "single_task_candidate_failures must be a positive integer"
        if not _is_number(value["relaunch_stop_fraction"]) or not 0 <= value["relaunch_stop_fraction"] <= 1:
            return "relaunch_stop_fraction must be between zero and one"
        return (
            None
            if value["candidate_attributable"] == ["killed_by_window", "invalid_session"]
            else "candidate_attributable must name the two executable candidate-attributable classes"
        )
    if name == "warmup":
        keys = {"salt", "k", "dataset_root_sha256"} | ({"repeats"} if native else set())
        if not isinstance(value, Mapping) or set(value) != keys:
            return "needs exactly salt, k, dataset_root_sha256"
        if native and (value["k"] != 1 or value["repeats"] != 16):
            return "native warm-up is one validation scene repeated 16 times initially"
        if not isinstance(value["salt"], str) or not value["salt"] or not _is_int(value["k"]) or value["k"] < 1:
            return "salt and a positive k are required"
        if not pinned.is_hex64(value["dataset_root_sha256"]):
            return "dataset_root_sha256 must be a SHA-256"
        if native:
            return None
        root = decisions["dataset_root_sha256"]["value"]
        return None if value["dataset_root_sha256"] == root else "differs from the sealed dataset_root_sha256"
    if name == "harness_commit_authorization":
        return (
            None
            if isinstance(value, Mapping) and set(value) == {"commit"} and pinned.is_hex40(value["commit"])
            else ("needs {commit: 40 hex}")
        )
    if name == "targets":
        if not isinstance(value, list) or not value or len(set(value)) != len(value) or set(value) - set(TARGETS):
            return f"must be a nonempty unique subset of {list(TARGETS)}"
        return None
    if name == "seeds":
        if (
            not isinstance(value, list)
            or not value
            or not all(_is_int(s) for s in value)
            or len(set(value)) != len(value)
        ):
            return "must be unique integer seeds"
        return None
    if name == "strict_comparison":
        return None if isinstance(value, bool) else "must be a boolean"
    if name == "confirmatory_analysis":
        if profile in DESCRIPTIVE_PROFILES:
            return (
                None
                if value == DESCRIPTIVE_ANALYSIS
                else "this profile requires descriptive_only with no confirmatory family"
            )
        if not isinstance(value, Mapping) or set(value) != _CONFIRMATORY_KEYS:
            return f"needs exactly {sorted(_CONFIRMATORY_KEYS)}"
        if profile == RULE8_NATIVE_PROFILE and (
            value["seeds"] != list(PLANNED_SEEDS) or value["draws"] != 10000 or value["alpha"] != 0.05
        ):
            return "Rule 8 seals all three planned seeds, 10,000 paired draws and 95% confidence intervals"
        if (value["a"], value["b"]) != CANDIDATES:
            return f"the family is a = {CANDIDATES[0]}, b = {CANDIDATES[1]}"
        if value["target"] not in TARGETS:
            return f"target must be one of {list(TARGETS)}"
        seeds = value["seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) < 2
            or not all(_is_int(seed) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            return "seeds must be at least two unique integer training seeds"
        if not _is_number(value["alpha"]) or not 0 < value["alpha"] < 1:
            return "alpha must lie in (0, 1)"
        for key in ("draws", "permutations"):
            if not _is_int(value[key]) or value[key] < 1:
                return f"{key} must be a positive integer"
        for key in ("bootstrap_seed", "permutation_seed", "latency_bootstrap_seed"):
            if not _is_int(value[key]) or value[key] < 0:
                return f"{key} must be a nonnegative integer"
        return None
    return f"no validator for {name}"


def decision_problem(
    name: str, value: Any, decisions: Mapping[str, Any], *, profile: str = STRICT_PROFILE
) -> str | None:
    """Why a sealed value of the decision ``name`` is invalid, or ``None`` (the validator of gate G1)."""
    return _decision_problem(name, value, decisions, profile=profile)


def _without_installed_rule_status(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """The aggregate without the ``status`` label that the installed censoring rule writes into its latency blocks.

    ``metrics.latency`` labels its own rule ``proposed_not_sealed`` until the code seals it; G1 binds the author's
    sealing decision and G5 binds it to the installed rule id, so that one label is not scanned as unsealed content.
    """
    from passage_eval.metrics import latency

    def strip(block: Any) -> Any:
        if (
            isinstance(block, Mapping)
            and block.get("rule") == latency.RULE_ID
            and block.get("status") == latency.RULE_STATUS
        ):
            return {key: value for key, value in block.items() if key != "status"}
        return block

    content = dict(aggregate)
    content["latency"] = strip(content.get("latency"))
    if isinstance(content.get("per_seed"), Mapping):
        content["per_seed"] = {
            seed: dict(entry, latency=strip(entry.get("latency"))) if isinstance(entry, Mapping) else entry
            for seed, entry in content["per_seed"].items()
        }
    return content


def _aggregate_problems(aggregate: Any, *, procedure: str, expected: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Problems of one rehearsal aggregate (``passage_eval.campaign`` output) of ``procedure`` for G2 and G5.

    ``expected`` gives the record's ``harness``, the sealed ``budget``, ``timing_profile`` and ``seeds``, the
    receipt's ``plan_sha256`` and the sealed ``min_warm_requests``. Every sealed seed is checked in ``per_seed``.
    """
    from passage_eval.metrics import latency

    if not isinstance(aggregate, Mapping) or aggregate.get("schema") != AGGREGATE_SCHEMA:
        return [("aggregate_schema_invalid", f"{AGGREGATE_SCHEMA} required")]
    problems = []
    markers = non_evidence_markers(aggregate)
    if markers or _flags_unqualified(aggregate):
        problems.append(("aggregate_evidence_not_real", f"markers {markers or ['qualified: false']}"))
    unsealed = unsealed_markers(_without_installed_rule_status(aggregate))
    if unsealed:
        problems.append(("aggregate_has_unsealed_values", unsealed))
    if aggregate.get("procedure") != procedure:
        problems.append(("aggregate_procedure_differs", f"the aggregate is of {aggregate.get('procedure')!r}"))
    if aggregate.get("target") != expected.get("target", HOST):
        code = "aggregate_not_host" if expected.get("target", HOST) == HOST else "aggregate_target_differs"
        problems.append((code, aggregate.get("target")))
    if aggregate.get("split") != "validation":
        problems.append(("aggregate_not_validation", aggregate.get("split")))
    if aggregate.get("harness") != expected["harness"]:
        problems.append(("aggregate_harness_differs", "aggregate made by another harness"))
    if expected["plan_sha256"] is None or aggregate.get("plan_sha256") != expected["plan_sha256"]:
        problems.append(("aggregate_plan_differs", "the aggregate is not of the rehearsal plan named by the receipt"))
    if aggregate.get("budget") != expected["budget"]:
        problems.append(("aggregate_budget_differs", "the aggregate budget is not the sealed budget"))
    if aggregate.get("timing_profile") != expected["timing_profile"]:
        problems.append(("aggregate_timing_profile_differs", aggregate.get("timing_profile")))
    population = aggregate.get("population")
    if not (
        isinstance(population, Mapping)
        and all(_is_int(population.get(key)) for key in ("n_source", "n_excluded", "n_admissible", "n_scored"))
        and population["n_admissible"] >= 1
        and population["n_source"] == population["n_excluded"] + population["n_admissible"]
        and population["n_scored"] == population["n_admissible"]
    ):
        problems.append(
            ("aggregate_population_invalid", "n_source = n_excluded + n_admissible = n_scored + n_excluded")
        )
    if not (
        pinned.is_hex64(aggregate.get("admissible_roster_sha256"))
        and pinned.is_hex64(aggregate.get("reference_admission_chain_sha256"))
    ):
        problems.append(("aggregate_rosters_missing", "admissible roster and admission chain digests"))
    receipts = aggregate.get("receipts")
    rows = [row for row in receipts if isinstance(row, Mapping)] if isinstance(receipts, list) else []
    # One completion per unit: a batch, or one of the halves the R7 rule split it into.
    units = [(row.get("batch_index"), row.get("unit")) for row in rows]
    if not (
        isinstance(receipts, list)
        and receipts
        and len(rows) == len(receipts)
        and all(_is_int(index) for index, _ in units)
        and len(set(units)) == len(units)
        and all(pinned.is_hex64(row.get("completion_sha256")) for row in rows)
    ):
        problems.append(("aggregate_receipts_missing", "one completion digest per batch or R7 unit"))
    equivalence = aggregate.get("canonical_equivalence")
    if not (isinstance(equivalence, Mapping) and equivalence.get("equal") is True):
        problems.append(("aggregate_not_equal_to_canonical_scorer", "per-batch metrics differ from the scorer"))
    seeds = expected["seeds"]
    per_seed = aggregate.get("per_seed")
    keys = [str(seed) for seed in seeds] if isinstance(seeds, list) else None
    if (
        keys is None
        or aggregate.get("seeds") != seeds
        or not isinstance(per_seed, Mapping)
        or sorted(per_seed) != sorted(keys)
    ):
        problems.append(("aggregate_seeds_differ", f"per_seed must report exactly the sealed seeds {seeds}"))
        return problems
    minimum = expected["min_warm_requests"]
    for key in keys:
        entry = per_seed[key] if isinstance(per_seed[key], Mapping) else {}
        seed_equivalence = entry.get("canonical_equivalence")
        if not (isinstance(seed_equivalence, Mapping) and seed_equivalence.get("equal") is True):
            problems.append(("aggregate_not_equal_to_canonical_scorer", f"seed {key}: metrics differ from the scorer"))
        block = entry.get("latency")
        if not isinstance(block, Mapping) or block.get("rule") != latency.RULE_ID:
            problems.append(("aggregate_latency_missing", f"seed {key}: no censored latency block"))
            continue
        n_warm = block.get("n_warm_requests")
        if block.get("meets_min_warm_requests") is not True or not _is_int(n_warm) or n_warm < minimum:
            problems.append(("aggregate_latency_below_minimum", f"seed {key}: {n_warm} warm request(s) < {minimum}"))
    return problems


# ----------------------------------------------------------------------------------------------- entry points


def _all_red(
    record_name: str, record_sha: str | None, evidence_dir: str | None, code: str, detail: Any
) -> AdmissionReport:
    reason = ({"code": code, "detail": str(detail)[:400]},)
    return AdmissionReport(
        record_name, record_sha, None, evidence_dir, None, tuple(GateResult(g, RED, reason) for g in GATES)
    )


def check_admission(
    record_path: str | Path,
    evidence_dir: str | Path | None = None,
    *,
    final_candidates: str | Path | None = None,
    now: _dt.datetime | None = None,
) -> AdmissionReport:
    """Evaluate the six gates of a ``passage-test-admission/v1`` record; never raises for a ``RED`` gate."""
    record_path = Path(record_path)
    evidence = Path(evidence_dir) if evidence_dir is not None else record_path.parent
    evidence_text = str(evidence)
    try:
        raw = pinned.read_bounded(record_path)
    except (pinned.PinnedFileError, OSError) as exc:
        return _all_red(record_path.name, None, evidence_text, "record_unreadable", exc)
    record_sha = pinned.sha256_bytes(raw)
    try:
        record = validate_record_structure(pinned.strict_json_loads(raw, name=record_path.name))
    except (pinned.PinnedFileError, AdmissionRecordError) as exc:
        return _all_red(record_path.name, record_sha, evidence_text, "record_structure_invalid", exc)
    if not evidence.is_dir():
        return _all_red(record_path.name, record_sha, evidence_text, "evidence_dir_missing", evidence_text)
    checker = _Checker(
        record,
        evidence,
        Path(final_candidates) if final_candidates is not None else _default_final_candidates(),
        now or _now(),
    )
    results = []
    for name in GATES:
        try:
            results.append(getattr(checker, name.lower())())
        except Exception as exc:  # a malformed value deep in the evidence must not crash the verdict
            results.append(
                GateResult(name, RED, ({"code": "gate_check_error", "detail": f"{type(exc).__name__}: {exc}"[:400]},))
            )
    return AdmissionReport(
        record_path.name,
        record_sha,
        record_core_sha256(record),
        evidence_text,
        record["split"],
        tuple(results),
        file_pins=tuple(sorted(checker.pins.items())),
        admitted=checker.admitted,
        external_pins=tuple(sorted(checker.external_pins.items())),
    )


def admission_allows(record_path: str | Path | None, evidence_dir: str | Path | None = None) -> bool:
    """Whether a record admits the test split (all six gates ``GREEN``); ``False`` without a record."""
    if record_path is None:
        return False
    return check_admission(record_path, evidence_dir).all_green


# ----------------------------------------------------------------------------------------------- the grant

_GRANT_TOKEN = object()


class AdmissionGrant:
    """Proof, inside one process, that a record checked all ``GREEN``; built only by ``grant_test_admission``.

    The guards accept the test split only with ``allow_test=True`` and a grant. A grant re-hashes its record, every
    evidence file it pinned, ``final-candidates.json`` and the installed ``final_decoders.py`` before each use
    (``ensure_unchanged``), so a record, evidence or frozen file edited after the check is refused. ``admits_roster``
    binds the grant to the sealed admissible roster and its derived children.
    """

    __slots__ = (
        "_token",
        "record_path",
        "evidence_dir",
        "record_sha256",
        "report",
        "_pins",
        "_external_pins",
        "_admissible",
    )

    def __init__(self, token, record_path: Path, evidence_dir: Path, report: AdmissionReport):
        if token is not _GRANT_TOKEN:
            raise TypeError("an AdmissionGrant is created only by grant_test_admission")
        if not report.all_green:
            raise TypeError("a grant needs an all-GREEN report")
        self._token = token
        self.record_path = record_path
        self.evidence_dir = evidence_dir
        self.record_sha256 = report.record_sha256
        self.report = report
        self._pins = dict(report.file_pins)
        self._external_pins = dict(report.external_pins)
        self._admissible = copy.deepcopy(report.admitted.get("admissible_roster"))

    @property
    def split(self) -> str:
        return self.report.split or ""

    @property
    def source_roster_sha256(self) -> str:
        return self.report.admitted["source_roster_sha256"]

    @property
    def admissible_roster_sha256(self) -> str:
        return self.report.admitted["admissible_roster_sha256"]

    def is_genuine(self) -> bool:
        return self._token is _GRANT_TOKEN and self.report.all_green

    def ensure_unchanged(self) -> None:
        """Raise ``ValueError`` when the record, a pinned evidence file or a frozen file changed since the check."""
        if pinned.sha256_bytes(pinned.read_bounded(self.record_path)) != self.record_sha256:
            raise ValueError("the admission record changed after it was checked")
        for relative, digest in self._pins.items():
            pinned.read_pinned(pinned.resolve_inside(self.evidence_dir, relative), digest)
        for path, digest in self._external_pins.items():
            pinned.read_pinned(Path(path), digest)

    def reference_ids(self) -> dict[str, str]:
        """The ``reference_id`` of every admissible task in the sealed reference inventory (gate G4)."""
        return dict(self.report.admitted.get("reference_ids") or {})

    def reference_policy(self) -> dict[str, Any]:
        """The sealed reference policy, cohort and search limit of the admission chain."""
        return dict(self.report.admitted.get("reference_policy") or {})

    def admits_roster(self, roster: Mapping[str, Any]) -> bool:
        """Whether ``roster`` is the sealed admissible roster or a child derived from it with the same blocks."""
        from passage_eval.io import roster as roster_io

        digest = roster.get("roster_sha256") if isinstance(roster, Mapping) else None
        if digest == self.admissible_roster_sha256:
            return True
        derivation = (roster.get("selection") or {}).get("derivation") if isinstance(roster, Mapping) else None
        if (
            not isinstance(derivation, Mapping)
            or derivation.get("parent_roster_sha256") != self.admissible_roster_sha256
        ):
            return False
        try:
            roster_io.verify_child_blocks(self._admissible, roster)
        except ValueError:
            return False
        return True


def grant_test_admission(
    record_path: str | Path,
    evidence_dir: str | Path | None = None,
    *,
    final_candidates: str | Path | None = None,
    refuse: Callable[[str], BaseException] | None = None,
) -> AdmissionGrant:
    """Check a record and return a grant when every gate is ``GREEN``; otherwise raise (``refuse(message)``)."""
    record_path = Path(record_path)
    evidence = Path(evidence_dir) if evidence_dir is not None else record_path.parent
    report = check_admission(record_path, evidence, final_candidates=final_candidates)
    if not report.all_green:
        reasons = "; ".join(
            f"{gate.gate}: {', '.join(r['code'] for r in gate.reasons[:3])}"
            for gate in report.gates
            if gate.status != GREEN
        )
        message = f"the test admission record {record_path.name!r} is not all GREEN ({reasons})"
        raise (refuse(message) if refuse is not None else PermissionError(message))
    return AdmissionGrant(_GRANT_TOKEN, record_path, evidence, report)


# ----------------------------------------------------------------------------------------------- preparation

PREPARATION_SCHEMA = "passage-test-preparation-approval/v1"
PREPARATION_OPERATOR_SCHEMA = "passage-test-preparation-operator-receipt/v1"
PREPARATION_SCOPE = "preparation"
#: Longest accepted time between a preparation approval and its expiry. A harness bound, not a protocol value.
PREPARATION_MAX_WINDOW = _dt.timedelta(hours=72)
_PREPARATION_KEYS = {
    "schema",
    "scope",
    "readers",
    "admission_record_sha256",
    "approver",
    "approved_at",
    "expires_at",
    "statement",
}
#: The RED reasons a preparation tolerates: the G4 and G5 artefacts that the preparation readers produce.
PREPARATION_ARTEFACT_REASONS = {
    "G4": frozenset(f"{name}_missing" for name in (*_REFERENCE_KEYS, "historical_roster")),
    "G5": frozenset({"warmup_roster_missing", "warmup_selection_missing"}),
}


def preparation_problems(report: AdmissionReport, record: Mapping[str, Any] | None) -> list[str]:
    """Why a checked record does not allow the preparation readers (empty: it allows them).

    Every RED reason must be a G4 or G5 artefact of ``PREPARATION_ARTEFACT_REASONS``, except one G1 reason: the
    ``population`` decision still ``required`` with a null value, because the roster builder inventories it.
    """
    population = None
    if isinstance(record, Mapping):
        population = ((record.get("protocol") or {}).get("decisions") or {}).get("population")
    pending_population = isinstance(population, Mapping) and population.get("status") == REQUIRED
    pending_population = pending_population and population.get("value") is None
    problems = []
    for gate in report.gates:
        tolerated = PREPARATION_ARTEFACT_REASONS.get(gate.gate, frozenset())
        for reason in gate.reasons:
            if reason["code"] in tolerated:
                continue
            if (
                gate.gate == "G1"
                and pending_population
                and reason["code"] == "decision_not_sealed"
                and reason["detail"] == f"population ({REQUIRED})"
            ):
                continue
            problems.append(f"{gate.gate}: {reason['code']}")
    decisions = ((record.get("protocol") or {}).get("decisions") or {}) if isinstance(record, Mapping) else {}
    selection = decisions.get("roster_selection")
    selection = selection.get("value") if isinstance(selection, Mapping) else None
    if not isinstance(selection, Mapping) or selection.get("n") != S_ROSTER_SIZE:
        problems.append(f"the preparation scope needs roster S with n={S_ROSTER_SIZE}")
    return problems


def _preparation_approval_problems(
    approval: Any, record_sha256: str | None, now: _dt.datetime, record: Mapping[str, Any] | None = None
) -> tuple[list[str], tuple[str, ...], _dt.datetime | None]:
    if isinstance(approval, Mapping) and approval.get("schema") == PREPARATION_OPERATOR_SCHEMA:
        from passage_eval.protocol.guards import PREPARATION_READERS

        problems = []
        expected = {
            "schema",
            "scope",
            "readers",
            "admission_record_sha256",
            "authorization_sha256",
            "operator",
            "created_at",
            "expires_at",
        }
        if set(approval) != expected or approval.get("scope") != PREPARATION_SCOPE:
            problems.append("operator receipt fields or scope differ")
        readers = approval.get("readers")
        if (
            not isinstance(readers, list)
            or not readers
            or not all(isinstance(item, str) for item in readers)
            or len(set(readers)) != len(readers)
            or not set(readers) <= set(PREPARATION_READERS)
        ):
            problems.append("operator receipt readers differ")
            readers = []
        existing = (record or {}).get("approval")
        authorization = existing.get("authorization") if isinstance(existing, Mapping) else None
        if not isinstance(authorization, Mapping) or approval.get("authorization_sha256") != pinned.canonical_sha256(
            authorization
        ):
            problems.append("operator receipt authorization differs")
        if record_sha256 is None or approval.get("admission_record_sha256") != record_sha256:
            problems.append("operator receipt names another record")
        if not isinstance(approval.get("operator"), str) or not approval["operator"].strip():
            problems.append("operator identity is required")
        expires = None
        try:
            created = parse_timestamp(approval.get("created_at"))
            expires = parse_timestamp(approval.get("expires_at"))
            if created > now or expires <= now or expires <= created or expires - created > PREPARATION_MAX_WINDOW:
                problems.append("operator receipt validity window is invalid")
        except (TypeError, ValueError):
            problems.append("operator receipt timestamps are invalid")
        return problems, tuple(readers), expires
    if not isinstance(approval, Mapping) or approval.get("schema") != PREPARATION_SCHEMA:
        return [f"the approval is not a {PREPARATION_SCHEMA} record"], (), None
    problems = []
    if set(approval) != _PREPARATION_KEYS:
        problems.append(f"approval keys must be {sorted(_PREPARATION_KEYS)}")
    from passage_eval.protocol.guards import PREPARATION_READERS

    if approval.get("scope") != PREPARATION_SCOPE:
        problems.append(f"the approval scope must be {PREPARATION_SCOPE!r}")
    readers = approval.get("readers")
    if (
        not isinstance(readers, list)
        or not readers
        or len(set(map(str, readers))) != len(readers)
        or not set(readers) <= set(PREPARATION_READERS)
    ):
        problems.append(f"readers must be a nonempty unique subset of {list(PREPARATION_READERS)}")
        readers = []
    if record_sha256 is None or approval.get("admission_record_sha256") != record_sha256:
        problems.append("the approval names another admission record (SHA-256 of the record file)")
    approver = approval.get("approver")
    if not isinstance(approver, str) or not approver.strip() or unsealed_markers(approver):
        problems.append("the approver must be named")
    statement = approval.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        problems.append("the approval needs the author's statement")
    expires = None
    try:
        approved = parse_timestamp(approval.get("approved_at"))
        expires = parse_timestamp(approval.get("expires_at"))
    except (TypeError, ValueError) as exc:
        problems.append(f"approval timestamps: {exc}")
    else:
        if approved > now:
            problems.append("the approval is dated in the future")
        if expires <= approved or expires - approved > PREPARATION_MAX_WINDOW:
            problems.append(f"the approval window must be positive and at most {PREPARATION_MAX_WINDOW}")
        if expires <= now:
            problems.append("the preparation approval has expired")
    return problems, tuple(readers), expires


class PreparationGrant:
    """Proof, inside one process, that a record allows the test preparation readers; built only by
    ``grant_test_preparation``.

    It never admits a roster for a run (``admits_roster`` is false), and the guards accept it only for a reader of
    ``guards.PREPARATION_READERS`` that the approval names. Before each use it re-hashes the record, the evidence
    files and frozen files of the check and the approval file, and refuses after the approval's expiry.
    """

    __slots__ = (
        "_token",
        "record_path",
        "evidence_dir",
        "approval_path",
        "approval_sha256",
        "record_sha256",
        "report",
        "readers",
        "expires_at",
        "_roster_selection",
        "_scope",
        "_pins",
        "_external_pins",
    )

    def __init__(
        self,
        token,
        record_path: Path,
        evidence_dir: Path,
        approval_path: Path,
        approval_sha256: str,
        report: AdmissionReport,
        readers: tuple[str, ...],
        expires_at: _dt.datetime,
        roster_selection: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
    ):
        if token is not _GRANT_TOKEN:
            raise TypeError("a PreparationGrant is created only by grant_test_preparation")
        self._token = token
        self.record_path = record_path
        self.evidence_dir = evidence_dir
        self.approval_path = approval_path
        self.approval_sha256 = approval_sha256
        self.record_sha256 = report.record_sha256
        self.report = report
        self.readers = tuple(readers)
        self.expires_at = expires_at
        self._roster_selection = dict(roster_selection or {})
        self._scope = dict(scope or {})
        self._pins = dict(report.file_pins)
        self._external_pins = dict(report.external_pins)

    @property
    def split(self) -> str:
        return self.report.split or ""

    def is_genuine(self) -> bool:
        return self._token is _GRANT_TOKEN and bool(self.readers)

    def ensure_unchanged(self) -> None:
        """Raise ``ValueError`` after expiry, or when the record, an evidence or frozen file or the approval changed."""
        if _now() >= self.expires_at:
            raise ValueError("the preparation approval has expired")
        if pinned.sha256_bytes(pinned.read_bounded(self.record_path)) != self.record_sha256:
            raise ValueError("the admission record changed after the preparation was approved")
        pinned.read_pinned(self.approval_path, self.approval_sha256)
        for relative, digest in self._pins.items():
            pinned.read_pinned(pinned.resolve_inside(self.evidence_dir, relative), digest)
        for path, digest in self._external_pins.items():
            pinned.read_pinned(Path(path), digest)

    def admits_roster(self, roster: Mapping[str, Any]) -> bool:
        """A preparation grant admits no roster for a run."""
        return False

    def require_roster_selection(self, *, rule: str, salt: str, n: int) -> None:
        """Refuse a test roster request outside the sealed, label-free S selection.

        This is deliberately usable only by the preparation roster reader.  It
        runs before the builder opens the pinned dataset receipt or a parquet
        file, so a caller cannot use a genuine preparation grant to probe a
        fixture-sized or otherwise arbitrary test roster.
        """
        from passage_eval.protocol.guards import TestSplitRefused

        expected = self._roster_selection
        if expected.get("n") != S_ROSTER_SIZE or n != S_ROSTER_SIZE or {"rule": rule, "salt": salt, "n": n} != expected:
            raise TestSplitRefused(
                f"the preparation grant permits only roster S (n={S_ROSTER_SIZE}) with its sealed selection rule and salt"
            )

    def require_roster_scope(self, *, rule: str, salt: str, n: int, cell: Mapping[str, Any], block_rule: str) -> None:
        """Check the request-only portion of the sealed preparation scope before a dataset is opened."""
        from passage_eval.protocol.guards import TestSplitRefused

        self.require_roster_selection(rule=rule, salt=salt, n=n)
        if cell != self._scope.get("evaluation_cell"):
            raise TestSplitRefused("the preparation grant permits only its sealed evaluation cell")
        blocks = self._scope.get("block_analysis")
        if not isinstance(blocks, Mapping) or block_rule != blocks.get("rule"):
            raise TestSplitRefused("the preparation grant permits only its sealed block selection rule")

    def require_dataset_scope(self, *, revision: Mapping[str, Any], root_sha256: str) -> None:
        """Bind the receipt just opened to the sealed revision and root before any parquet is opened."""
        from passage_eval.protocol.guards import TestSplitRefused

        if revision != self._scope.get("dataset_revision") or root_sha256 != self._scope.get("dataset_root_sha256"):
            raise TestSplitRefused("the preparation grant permits only its sealed dataset revision and root")

    def require_roster_scope_from_roster(self, roster: Mapping[str, Any]) -> None:
        """Check a prepared reader's roster against every sealed S-scope binding."""
        dataset = roster.get("dataset") if isinstance(roster, Mapping) else None
        selection = roster.get("selection") if isinstance(roster, Mapping) else None
        root = roster.get("dataset_root_identity") if isinstance(roster, Mapping) else None
        blocks = roster.get("geographic_blocks") if isinstance(roster, Mapping) else None
        if not all(isinstance(value, Mapping) for value in (dataset, selection, root, blocks)):
            from passage_eval.protocol.guards import TestSplitRefused

            raise TestSplitRefused("the preparation grant needs a complete roster scope")
        self.require_roster_scope(
            rule=selection.get("key"),
            salt=selection.get("salt"),
            n=selection.get("n_selected"),
            cell=roster.get("cell"),
            block_rule=blocks.get("rule"),
        )
        self.require_dataset_scope(
            revision={key: dataset.get(key) for key in ("repo", "commit", "config", "resolution")},
            root_sha256=root.get("sha256"),
        )


def grant_test_preparation(
    record_path: str | Path,
    approval_path: str | Path,
    evidence_dir: str | Path | None = None,
    *,
    final_candidates: str | Path | None = None,
    now: _dt.datetime | None = None,
    refuse: Callable[[str], BaseException] | None = None,
) -> PreparationGrant:
    """Check a record and a preparation approval; return a ``PreparationGrant`` or raise (``refuse(message)``).

    The record's only RED reasons must be the G4/G5 artefacts of ``PREPARATION_ARTEFACT_REASONS`` (and the pending
    ``population`` decision); the approval (``PREPARATION_SCHEMA``) must name the record file's SHA-256, a nonempty
    subset of the preparation readers, the approver, a statement and a window of at most
    ``PREPARATION_MAX_WINDOW`` that contains the present time.
    """
    record_path, approval_path = Path(record_path), Path(approval_path)
    evidence = Path(evidence_dir) if evidence_dir is not None else record_path.parent
    moment = now or _now()
    report = check_admission(record_path, evidence, final_candidates=final_candidates, now=moment)
    try:
        record = pinned.read_pinned_json(record_path, report.record_sha256)
    except (pinned.PinnedFileError, OSError):
        record = None
    problems = preparation_problems(report, record)
    try:
        raw = pinned.read_bounded(approval_path)
        approval = pinned.strict_json_loads(raw, name=approval_path.name)
    except (pinned.PinnedFileError, OSError) as exc:
        approval, raw = None, b""
        problems.append(f"approval unreadable: {exc}")
    approval_problems, readers, expires = _preparation_approval_problems(approval, report.record_sha256, moment, record)
    problems += approval_problems if approval is not None else []
    if problems:
        message = f"the test preparation of {record_path.name!r} is refused ({'; '.join(problems[:6])})"
        raise (refuse(message) if refuse is not None else PermissionError(message))
    return PreparationGrant(
        _GRANT_TOKEN,
        record_path,
        evidence,
        approval_path,
        pinned.sha256_bytes(raw),
        report,
        readers,
        expires,
        ((record.get("protocol") or {}).get("decisions") or {}).get("roster_selection", {}).get("value", {}),
        {
            name: ((record.get("protocol") or {}).get("decisions") or {}).get(name, {}).get("value")
            for name in ("dataset_revision", "dataset_root_sha256", "evaluation_cell", "block_analysis")
        },
    )


def preparation_approval_template(record_sha256: str, readers=None) -> dict[str, Any]:
    """An unfilled preparation approval for ``record_sha256``: the approver, times and statement are the author's."""
    from passage_eval.protocol.guards import PREPARATION_READERS

    return {
        "schema": PREPARATION_SCHEMA,
        "scope": PREPARATION_SCOPE,
        "readers": list(PREPARATION_READERS if readers is None else readers),
        "admission_record_sha256": record_sha256,
        "approver": None,
        "approved_at": None,
        "expires_at": None,
        "statement": None,
    }


def template_record(record_id: str = "protocol-256-sealing-record") -> dict[str, Any]:
    """An unsealed draft record: every author decision is listed as ``required`` with a null value."""
    return {
        "schema": SCHEMA,
        "status": "draft",
        "split": "test",
        "record_id": record_id,
        "protocol": {
            "version": "required",
            "document": None,
            "decisions": {name: {"value": None, "status": REQUIRED, "note": text} for name, text in DECISIONS.items()},
        },
        "harness": {"commit": None, "source_sha256": None},
        "candidates": {
            name: {
                "checkpoint_sha256": None,
                "predictor": None,
                "decoder": None,
                "decoder_tree_sha256": None,
                "final_decoders_sha256": None,
                "targets": {
                    target: {"bundle": None, "bundle_sha256": None, "engine_sha256": None, "container_digest": None}
                    for target in TARGETS
                },
            }
            for name in CANDIDATES
        },
        "evidence": {
            "native_chain_rehearsal": None,
            "edge": {target: None for target in EDGE_TARGETS},
            "references": {name: None for name in _REFERENCE_KEYS},
            "warmup": {name: None for name in _WARMUP_KEYS},
        },
        "approval": None,
        "notes": "Draft template: every decision is required and unsealed; no evidence is attached.",
    }


__all__ = [
    "SCHEMA",
    "GATES",
    "GATE_TITLES",
    "GREEN",
    "RED",
    "CANONICAL_PLANNER_ENTRIES",
    "DECISIONS",
    "EDGE_TARGETS",
    "FROZEN_FINAL_DECODERS_SHA256",
    "HARNESS_TRACKED_PATHS",
    "TARGETS",
    "AdmissionGrant",
    "AdmissionReport",
    "GateResult",
    "PREPARATION_ARTEFACT_REASONS",
    "PREPARATION_MAX_WINDOW",
    "PREPARATION_SCHEMA",
    "PreparationGrant",
    "admission_allows",
    "check_admission",
    "committed_final_candidates",
    "decision_problem",
    "decoder_tree_sha256",
    "format_report",
    "grant_test_admission",
    "grant_test_preparation",
    "preparation_approval_template",
    "preparation_problems",
    "non_evidence_markers",
    "record_core_sha256",
    "template_record",
    "unsealed_markers",
    "validate_record_structure",
]
