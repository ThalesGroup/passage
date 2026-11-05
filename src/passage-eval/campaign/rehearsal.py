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

"""Build the ``passage-native-chain-rehearsal/v1`` receipt of a validation rehearsal from its real campaign outputs.

``build_rehearsal_receipt`` reads a completed campaign of a sealed validation plan on the host and the campaign
aggregates of its procedures, which must lie under the evidence directory of the admission record. It refuses to
write a receipt unless every property that gate G2 of ``passage-test-admission/v1`` checks holds in the outputs
themselves:

* the plan is a validation plan on the host, read with its pin, and the running harness is the plan's commit and
  source hash with a clean working tree (``protocol.admission.HARNESS_TRACKED_PATHS``);
* every aggregate file equals the aggregate recomputed now from the campaign outputs (``aggregate_campaign``), is
  equal to the canonical scorer for every seed, and has at least the plan's minimum of warm requests; the plan's
  procedures include Seg and Reg, and every procedure has exactly one aggregate;
* the warm-up was declared and performed in every session, and at least one re-warm happened;
* a deliberate kill was exercised: at least one ``deliberate_kill`` entry of ``relaunch-log.jsonl`` whose decision is
  a planned R7 relaunch, and no archived attempt is scored;
* a resume was exercised: an invocation paused at a batch boundary (``run --stop-after-batches``) and a later one
  resumed at least one completed batch; no unplanned relaunch happened (no other R7 entry, no operator relaunch).

The receipt names the plan, the harness, the candidates' frozen checkpoints, the budget, the timing profile, the
warm-up roster, the counts above and one ``{path, sha256}`` reference per aggregate, relative to the evidence
directory. Whether the budget, profile and procedures are the *sealed* ones is decided by gate G2 against the record.
Hashes prove byte correspondence, not authorship.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from passage_eval.campaign import REHEARSAL_RECEIPT_SCHEMA
from passage_eval.io import pinned

CANDIDATE_PROCEDURES = ("seg", "reg")


class RehearsalNotQualified(ValueError):
    """The campaign outputs do not show every property a native chain rehearsal must show; no receipt is written."""


def _resume(output: Path) -> dict[str, Any]:
    from passage_eval.campaign.runner import INVOCATIONS_FILE, _read_jsonl

    invocations = _read_jsonl(output / INVOCATIONS_FILE)
    paused = [index for index, row in enumerate(invocations) if row.get("outcome") == "paused"]
    resumed = [
        len(row.get("resumed_batches") or [])
        for index, row in enumerate(invocations)
        if paused and index > paused[0] and row.get("outcome") == "complete"
    ]
    return {
        "exercised": bool(paused) and any(count >= 1 for count in resumed),
        "n_invocations": len(invocations),
        "n_batches_resumed": max(resumed, default=0),
    }


def build_rehearsal_receipt(
    plan_path: str | Path,
    plan_sha256: str,
    *,
    evidence_dir: str | Path,
    aggregates: Mapping[str, str],
) -> dict[str, Any]:
    """The receipt of a qualifying validation rehearsal; raise ``RehearsalNotQualified`` with every problem otherwise.

    ``aggregates`` maps each procedure of the plan to its aggregate file, relative to ``evidence_dir``.
    """
    from passage_eval.campaign.aggregate import aggregate_campaign
    from passage_eval.campaign.plan import read_plan
    from passage_eval.campaign.runner import DELIBERATE, RELAUNCH_LOG, _read_jsonl
    from passage_eval.protocol.admission import _uncommitted_harness_paths

    campaign = read_plan(plan_path, plan_sha256)  # validation only: no admission grant is given
    plan = campaign.plan
    problems: list[str] = []
    if plan["split"] != "validation":
        problems.append("the rehearsal plan is not a validation plan")
    if plan["target"] != "host":
        problems.append("the rehearsal plan does not run on the host")
    dirty = _uncommitted_harness_paths()
    if dirty is None:
        problems.append("the running harness is not a git checkout")
    elif dirty:
        problems.append(f"the running harness tree is not clean ({len(dirty)} path(s), e.g. {dirty[:3]})")
    missing = [name for name in CANDIDATE_PROCEDURES if name not in plan["procedures"]]
    if missing:
        problems.append(f"the rehearsal plan does not run {missing}")
    if sorted(aggregates) != sorted(plan["procedures"]):
        problems.append(f"one aggregate per procedure is required: {sorted(plan['procedures'])}")
    evidence = Path(evidence_dir)
    output = Path(plan["output_dir"])
    references, warm, rewarms, latency = {}, True, 0, {}
    archived_scored = 0
    for name in sorted(set(aggregates) & set(plan["procedures"])):
        try:
            location = pinned.resolve_inside(evidence, aggregates[name])
            raw = pinned.read_bounded(location)
            stored = pinned.strict_json_loads(raw, name=aggregates[name])
        except (pinned.PinnedFileError, OSError) as exc:
            problems.append(f"{name}: aggregate unreadable under the evidence directory ({exc})")
            continue
        recomputed = aggregate_campaign(plan_path, plan_sha256, procedure=name)
        if stored != recomputed:
            problems.append(f"{name}: the aggregate file differs from the aggregate recomputed from the campaign")
            continue
        references[name] = {
            "aggregate": {"path": str(pinned.safe_relative(aggregates[name])), "sha256": pinned.sha256_bytes(raw)}
        }
        if stored["canonical_equivalence"]["equal"] is not True:
            problems.append(f"{name}: the aggregate is not equal to the canonical scorer")
        for seed, entry in stored["per_seed"].items():
            block = entry["latency"]
            latency[f"{name}/{seed}"] = {
                "n_warm_requests": block["n_warm_requests"],
                "meets_min_warm_requests": block["meets_min_warm_requests"],
            }
            if block["meets_min_warm_requests"] is not True:
                problems.append(f"{name}, seed {seed}: fewer warm requests than the plan's minimum")
            warmup = block.get("warmup") or {}
            warm = warm and warmup.get("declared") is True and warmup.get("performed") is True
        rewarms += stored["restarts"]["rewarms"]
        completed = {row["completion_sha256"] for row in stored["receipts"]}
        for archived in (output / "relaunched").glob("*/completion.json"):
            archived_scored += pinned.sha256_bytes(pinned.read_bounded(archived)) in completed
    if not warm:
        problems.append("the warm-up was not declared and performed in every session")
    if rewarms < 1:
        problems.append("no re-warm was exercised")
    log = _read_jsonl(output / RELAUNCH_LOG)
    deliberate = [row for row in log if row.get("kind") == "r7" and row.get("classification") == DELIBERATE]
    planned = [row for row in deliberate if row.get("decision") == "relaunch"]
    unplanned = [row for row in log if row not in deliberate]
    if not deliberate or len(planned) < len(deliberate):
        problems.append("no deliberately killed batch was relaunched under the planned R7 rule")
    if archived_scored:
        problems.append(f"{archived_scored} archived attempt(s) are scored")
    resume = _resume(output)
    if not resume["exercised"]:
        problems.append("no resume after a pause at a batch boundary was exercised")
    if unplanned:
        problems.append(f"{len(unplanned)} unplanned relaunch(es) happened")
    if problems:
        raise RehearsalNotQualified("the rehearsal does not qualify: " + "; ".join(problems))
    return {
        "schema": REHEARSAL_RECEIPT_SCHEMA,
        "split": plan["split"],
        "target": plan["target"],
        "plan_sha256": plan_sha256,
        "campaign_id": plan["campaign_id"],
        "harness": {"commit": plan["harness"]["commit"], "source_sha256": plan["harness"]["source_sha256"]},
        "candidates": {
            name: {"checkpoint_sha256": plan["procedures"][name]["checkpoint_sha256"]} for name in CANDIDATE_PROCEDURES
        },
        "budget": plan["budget"],
        "timing_profile": plan["timing"]["profile"],
        "warmup": {
            "performed": warm,
            "roster_sha256": plan["warmup"]["roster"]["roster_sha256"],
            "k": plan["warmup"]["k"],
        },
        "rewarm": {"exercised": rewarms >= 1, "n_rewarm_events": rewarms},
        "deliberate_kill": {
            "exercised": True,
            "n_deliberate_kills": len(deliberate),
            "n_planned_relaunches": len(planned),
            "killed_attempts_scored": archived_scored,
            "units": sorted({row["unit"] for row in deliberate}),
        },
        "resume": {
            "exercised": True,
            "n_batches_resumed": resume["n_batches_resumed"],
            "n_unplanned_relaunches": len(unplanned),
        },
        "latency": {"min_warm_requests": plan["aggregation"]["min_warm_requests"], "rows": latency},
        "procedures": references,
        "verified": False,
    }
