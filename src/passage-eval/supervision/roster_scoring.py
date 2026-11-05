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

"""Roster-aware offline scorer for supervised runs: ``python -I -m passage_eval.supervision score-roster``.

``passage_eval.cli.cmd_score`` reloads the scenes from the recorded spec. A roster run selects its rows through a
salted, label-free ranking over the whole population (:mod:`passage_eval.io.roster`), which the spec does not carry, so
``cmd_score`` cannot reproduce them. This scorer runs in its own process, on the scorer's side of the trust boundary
(it reads the dataset and the labels), and does the following:

1. validates the pinned roster (digest equal to the operator's pin) and every public run manifest against it:
   roster id, digest, ``n_expected``, the pinned dataset root identity, the dataset repository, revision, split and
   resolution, the cell (cost model, weight, task variant), the exact ordered task list, and the ordinary input identity
   of every task (the fields the roster recorded, as the supervisor wrote them into ``request_specs``);
2. loads exactly the roster's rows with :func:`passage_eval.io.roster.load_roster_scenes` (source files re-hashed,
   ``PINNED.json`` checked against the roster) and checks the dataset content hash and the reference identity that the
   supervisor recorded against what is recomputed here;
3. copies the run to a private directory, attaches the re-qualified references to the copy of each manifest (the same
   staged layout as :func:`passage_eval.supervision.staging.stage_for_scoring`, with the public manifest kept beside it)
   and calls :func:`passage_eval.execution.score_run_artifact` with the roster's block table
   (:func:`passage_eval.io.roster.roster_block_table`) and :func:`passage_eval.execution.write_scored_artifact`. The
   per-sample ``cluster_id`` then holds the roster's blocks (a derived roster's inherited parent blocks), labelled
   ``sealed_roster_geographic_block`` only for a ``passage-roster-v2`` roster that binds the installed block rule and
   ``unsealed_roster_geographic_block`` for a ``passage-roster-v1`` roster. Tasks without footprint are in the
   sealed ``unplaced`` block when the roster declares it; otherwise a roster with an unplaced task leaves every
   block unknown.

Under the independent analytic reference policy the rows are loaded with the tolerant execution loader (a defective
published label is a dataset finding, never the reference). For an admitted test roster, and whenever
``reference_inventory`` is given, every re-qualified ``reference_id`` must be the sealed inventory's.

Every roster task must have a record in every pass; a run that lacks one is refused and nothing is left behind. The scored
directory holds reference routes and costs: keep it private and publish only a separately produced projection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from passage_eval.supervision import READABLE_TIMING_BOUNDARY_IDS
from passage_eval.supervision.staging import StagingError, attach_references

SCORER = "passage_eval.supervision score-roster"


class RosterScoringError(StagingError):
    """The run cannot be scored against this roster; nothing was written."""


def _fail(message: str) -> None:
    raise RosterScoringError(message)


def _read(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = path.read_bytes()
    manifest = json.loads(data)
    if not isinstance(manifest, dict):
        _fail(f"{path}: the run manifest is not an object")
    return data, manifest


def executed_task_ids(manifest: dict[str, Any], task_ids: list[str]) -> list[str]:
    """The roster tasks a run must cover: every task, or the measured prefix of a campaign session.

    A campaign session (``supervision.session``) closed by dynamic admission measures a prefix of its roster and
    defers the rest, in order, to the next window. Its manifest must declare exactly that prefix and that remainder;
    anything else (a gap, a reordering, a deferred task that was measured, an invalid session) is refused.
    """
    from passage_eval.supervision import session as session_mod

    supervision = manifest.get("supervision") if isinstance(manifest.get("supervision"), dict) else {}
    block = supervision.get("session")
    if block is None:
        return task_ids
    if not isinstance(block, dict) or block.get("policy") != session_mod.SESSION_POLICY_ID:
        _fail("the run declares an unknown session policy")
    executed, deferred = block.get("executed_sample_ids"), block.get("deferred_sample_ids")
    n = block.get("n_executed")
    if (
        block.get("status") not in (session_mod.COMPLETE, session_mod.CLOSED)
        or not isinstance(executed, list)
        or not isinstance(deferred, list)
        or isinstance(n, bool)
        or not isinstance(n, int)
        or not 1 <= n <= len(task_ids)
        or executed != task_ids[:n]
        or deferred != task_ids[n:]
        or (block["status"] == session_mod.COMPLETE) != (n == len(task_ids))
        or block.get("n_tasks") != len(task_ids)
    ):
        _fail("the session does not declare a measured prefix of the roster and its deferred remainder, in order")
    return executed


def check_public_manifest(manifest: dict[str, Any], payload: dict[str, Any], roster_id: str, spec) -> None:
    """Everything about a public roster-run manifest that can be checked against the roster alone (no dataset read)."""
    from passage_eval.io.roster import IDENTITY_FIELDS
    from passage_eval.supervision.supervisor import _roster_spec_mismatches

    supervision = manifest.get("supervision")
    if (
        manifest.get("reference_qualification_withheld") is not True
        or "reference_qualification" in manifest
        or not isinstance(supervision, dict)
        or supervision.get("schema") is None
    ):
        _fail("not a supervised public manifest (references must be withheld)")
    if manifest.get("timing_boundary") not in READABLE_TIMING_BOUNDARY_IDS:
        _fail(f"unknown timing boundary {manifest.get('timing_boundary')!r}")
    block = supervision.get("roster")
    if not isinstance(block, dict):
        _fail("the run is not a roster run (no supervision.roster block); score it with cmd_score or stage-score")
    expected_n = payload["selection"]["n_expected"]
    if block.get("roster_sha256") != payload["roster_sha256"]:
        _fail("the run was made against a different roster (roster_sha256 differs)")
    if block.get("roster_id") != roster_id:
        _fail(f"the run records roster id {block.get('roster_id')!r}, expected {roster_id!r}")
    if block.get("n_expected") != expected_n or block.get("n_loaded") != expected_n:
        _fail(
            f"the run records n_expected={block.get('n_expected')} n_loaded={block.get('n_loaded')}, "
            f"the roster expects {expected_n}"
        )
    if block.get("dataset_root_identity") != payload["dataset_root_identity"]:
        _fail("the run was made against another dataset root identity than the roster pins")
    dataset = manifest.get("dataset") or {}
    wanted = payload["dataset"]
    if (
        dataset.get("repo") != wanted["repo"]
        or dataset.get("revision") != wanted["commit"]
        or dataset.get("split") != wanted["split"]
        or dataset.get("resolution") != wanted["resolution"]
        or dataset.get("task_variant") != payload["cell"]["task_variant"]
    ):
        _fail(
            "the run's dataset identity (repository, revision, split, resolution, task variant) differs from the roster's"
        )
    mismatches = _roster_spec_mismatches(payload, spec)
    if mismatches:
        _fail("the run's spec does not match the roster's cell: " + "; ".join(mismatches))
    all_ids = [task["sample_id"] for task in payload["tasks"]]
    task_ids = executed_task_ids(manifest, all_ids)
    run_ids = manifest.get("roster")
    if run_ids != task_ids:
        got = set(run_ids) if isinstance(run_ids, list) else set()
        _fail(
            "the run does not cover every roster task in roster order "
            f"({len(set(task_ids) - got)} roster tasks missing, {len(got - set(task_ids))} not in the roster, "
            f"{len(run_ids) if isinstance(run_ids, list) else 'no'} listed of {len(task_ids)})"
        )
    specs = manifest.get("request_specs")
    if not isinstance(specs, dict) or set(specs) != set(task_ids):
        _fail("the run's request specifications do not align with the roster")
    for task in payload["tasks"][: len(task_ids)]:
        differing = sorted(
            field for field in IDENTITY_FIELDS if specs[task["sample_id"]].get(field) != task["input_identity"][field]
        )
        if differing:
            _fail(
                f"{task['sample_id']}: the run recorded another ordinary input identity than the roster in {differing}"
            )
    procedure, identity = manifest.get("procedure"), manifest.get("candidate_identity")
    if (
        not isinstance(procedure, dict)
        or not isinstance(identity, dict)
        or procedure.get("candidate_identity") != identity
        or identity.get("schema") != "passage-candidate-identity-v1"
        or hashlib.sha256(json.dumps(procedure, sort_keys=True).encode()).hexdigest() != manifest.get("procedure_id")
    ):
        _fail("persisted candidate/procedure identity is missing or inconsistent")


def score_roster_run(
    run: str | Path,
    roster_path: str | Path,
    roster_sha256: str,
    dataset_root: str | Path,
    output: str | Path,
    *,
    roster_id: str | None = None,
    verify_file_hashes: bool = True,
    allow_test: bool = False,
    admission: Any = None,
    reference_inventory: Any = None,
) -> list[Path]:
    """Score every series of a supervised roster run into a new private directory; returns the scored directories.

    A roster of the test split raises ``TestSplitRefused`` before any run or dataset file is read, unless
    ``allow_test`` is true and ``admission`` is an all-GREEN test admission grant (``guards.test_split_allowed``)
    that admits this roster (the sealed admissible roster or a child derived from it). ``reference_inventory`` (a
    ``passage-reference-admission/v1`` inventory) binds the re-qualified references of a validation run as well.
    """
    from passage_eval import cli
    from passage_eval.execution import read_run_artifact, score_run_artifact, write_scored_artifact
    from passage_eval.io import roster as roster_io
    from passage_eval.protocol.guards import refuse_test_roster

    payload = roster_io.load_roster(roster_path, expected_sha256=roster_sha256)
    refuse_test_roster(  # before any label is read
        payload, allow_test=allow_test, admission=admission, context="score-roster", require_admitted_roster=True
    )
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"scoring output already exists: {output}")
    if output.is_relative_to(run) or run.is_relative_to(output):
        _fail("the scoring output and the run directory must not contain one another")
    manifests = sorted(run.rglob("run_manifest.json"))
    if not manifests:
        _fail(f"no run artifacts found in {run}")
    expected_id = roster_id or payload["benchmark_id"]
    specs = []
    for public in manifests:  # cheap checks first: no label is read for a bad run
        _, manifest = _read(public)
        spec = cli.BenchSpec(**manifest["bench_spec"])
        check_public_manifest(manifest, payload, expected_id, spec)
        specs.append(spec)
    policies = {(spec.reference_policy, spec.reference_cohort_id, spec.reference_max_expansions) for spec in specs}
    if len(policies) != 1:
        _fail("the run's series use different reference policies, cohorts or search limits")
    scenes = roster_io.load_roster_scenes(
        payload,
        dataset_root,
        verify_file_hashes=verify_file_hashes,
        tolerant_labels=specs[0].reference_policy == "independent_analytic_reference",
    )
    if [scene.sample_id for scene in scenes] != [task["sample_id"] for task in payload["tasks"]]:
        _fail("the loaded scenes do not cover every roster task in order")
    content_hash = cli._dataset_content_hash(scenes)
    qualification = cli._qualification(scenes, specs[0])
    from passage_eval.protocol.reference_admission import (
        ReferenceIdentityMismatch,
        check_reference_ids,
        inventory_reference_ids,
    )

    expected_ids = []
    if admission is not None and payload["dataset"]["split"] == "test":
        expected_ids.append(admission.reference_ids())
    if reference_inventory is not None:
        expected_ids.append(inventory_reference_ids(reference_inventory))
    for expected in expected_ids:
        try:
            check_reference_ids(qualification, expected, context="score-roster")
        except ReferenceIdentityMismatch as exc:
            _fail(str(exc))
    reference_identity = hashlib.sha256(json.dumps(qualification, sort_keys=True).encode()).hexdigest()
    for public in manifests:
        manifest = json.loads(public.read_bytes())
        if manifest["dataset"]["content_hash"] != content_hash:
            _fail("dataset content differs from the supervised run")
        if manifest["reference_identity"] != reference_identity:
            _fail("re-qualified references differ from the identity recorded by the supervisor")
    block_table = roster_io.roster_block_table(payload)  # the roster's blocks, never request metadata
    shutil.copytree(run, output)
    scored: list[Path] = []
    try:
        for public_path in sorted(output.rglob("run_manifest.json")):
            public_bytes, manifest = _read(public_path)
            artifact = read_run_artifact(public_path.parent)  # artifact hashes, structure, complete unique records
            attach_references(
                public_path,
                public_bytes,
                manifest,
                bench_spec=manifest["bench_spec"],
                dataset=manifest["dataset"],
                qualification=qualification,
                records=len(artifact.records),
                stage_extra={
                    "scorer": SCORER,
                    "roster": {
                        "roster_id": expected_id,
                        "roster_sha256": payload["roster_sha256"],
                        "n_expected": payload["selection"]["n_expected"],
                    },
                },
            )
            staged = read_run_artifact(public_path.parent)  # re-read with the attached references, as cmd_score does
            measured = set(executed_task_ids(manifest, [task["sample_id"] for task in payload["tasks"]]))
            run_scenes = [scene for scene in scenes if scene.sample_id in measured]  # a session's measured prefix
            metrics, arrays = score_run_artifact(staged, run_scenes, geographic_blocks=block_table)
            write_scored_artifact(staged, metrics, arrays)
            scored.append(public_path.parent)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)  # no half-scored directory is left behind
        raise
    return scored
