# MIT License
#
# Copyright (c) 2026 THALES

"""Scorer-side staging: re-attach the references that the supervisor withheld.

A supervised run directory is safe to move around: its manifests carry no
reference path, cost or dataset location (``reference_qualification_withheld``).
``passage_eval.cli.cmd_score`` however scores from ``manifest.reference_qualification``.
This step runs in the *scorer* process, with access to the dataset, and produces a
private copy that ``cmd_score`` accepts unchanged.

The staging refuses to proceed unless, for every manifest:

* the dataset content hash recomputed here equals the recorded one;
* the references re-qualified here hash to the recorded ``reference_identity``;
* the recorded (public) ``bench_spec`` equals the private spec with only
  ``dataset_path`` removed.

The original public manifest is kept beside the staged one as
``public_run_manifest.json`` and its SHA-256 is recorded in ``scoring_stage``.
The staged directory contains reference paths and costs: keep it private.

Runs written under either timing boundary (``v2-supervised-ipc`` or ``v3-supervised-ipc-fresh-pass``) can be
staged; a run with another boundary is refused. A roster run must be staged with its roster (``roster=``): the
scenes come from the pinned roster rows, and the roster digest must equal the one recorded in the manifest.
``passage_eval.cli.cmd_score`` reloads scenes from the spec, not from a roster, so it cannot score such a run;
:mod:`passage_eval.supervision.roster_scoring` (``score-roster``) does, and shares :func:`attach_references`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from passage_eval.execution.artifacts import json_safe, read_run_artifact
from passage_eval.supervision import READABLE_TIMING_BOUNDARY_IDS

STAGE_SCHEMA = "passage-scoring-stage-v1"


class StagingError(ValueError):
    pass


def _normalized(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True))


def attach_references(
    public_path: Path,
    public_bytes: bytes,
    manifest: dict,
    *,
    bench_spec: dict,
    dataset: dict,
    qualification: Any,
    records: int,
    stage_extra: dict | None = None,
) -> None:
    """Replace a copied public manifest by the staged one: references attached, the public bytes kept beside it.

    ``public_path`` is the manifest inside the private copy. The staged manifest differs from the public one in exactly
    the keys the leaderboard side expects (``bench_spec``, ``dataset``, ``reference_qualification_withheld``,
    ``reference_qualification`` and ``scoring_stage``); everything else is unchanged.
    """
    (public_path.parent / "public_run_manifest.json").write_bytes(public_bytes)
    staged_manifest = dict(manifest)
    staged_manifest.pop("reference_qualification_withheld")
    staged_manifest["bench_spec"] = bench_spec
    staged_manifest["dataset"] = dataset
    staged_manifest["reference_qualification"] = json_safe(qualification)
    staged_manifest["scoring_stage"] = {
        "schema": STAGE_SCHEMA,
        "public_manifest_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "records": records,
        **(stage_extra or {}),
    }
    public_path.write_text(json.dumps(json_safe(staged_manifest), indent=2, sort_keys=True, allow_nan=False) + "\n")


def stage_for_scoring(run: str | Path, spec, output: str | Path, *, roster=None) -> list[Path]:
    from passage_eval import cli

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"staging output already exists: {output}")
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise StagingError("staging output and run directory must not contain one another")
    manifests = sorted(run.rglob("run_manifest.json"))
    if not manifests:
        raise StagingError(f"no run artifacts found in {run}")
    cli._validate_spec(spec)
    for public in manifests:
        recorded = json.loads(public.read_bytes()).get("supervision", {}).get("roster")
        if (recorded is None) != (roster is None) or (
            roster is not None and recorded["roster_sha256"] != roster.sha256
        ):
            raise StagingError("the run and the staging request disagree about the roster")
    if roster is None:
        scenes = cli._load_scenes(spec)
    else:
        from passage_eval.supervision.supervisor import _load_roster_scenes

        _, scenes = _load_roster_scenes(roster, spec)
    content_hash = cli._dataset_content_hash(scenes)
    qualification = cli._qualification(scenes, spec)
    reference_identity = hashlib.sha256(json.dumps(qualification, sort_keys=True).encode()).hexdigest()
    shutil.copytree(run, output)
    staged = []
    for public_path in sorted(output.rglob("run_manifest.json")):
        public_bytes = public_path.read_bytes()
        manifest = json.loads(public_bytes)
        artifact = read_run_artifact(public_path.parent)  # hashes and structure of the run data
        if (
            manifest.get("reference_qualification_withheld") is not True
            or "reference_qualification" in manifest
            or manifest.get("supervision", {}).get("schema") is None
        ):
            raise StagingError("not a supervised public manifest (references must be withheld)")
        if manifest.get("timing_boundary") not in READABLE_TIMING_BOUNDARY_IDS:
            raise StagingError(f"unknown timing boundary {manifest.get('timing_boundary')!r}")
        if manifest["dataset"]["content_hash"] != content_hash:
            raise StagingError("dataset content differs from the supervised run")
        if manifest["reference_identity"] != reference_identity:
            raise StagingError("re-qualified references differ from the identity recorded by the supervisor")
        seeded = cli._seed_spec(spec, int(manifest["seed"]))
        expected = _normalized(seeded.to_dict())
        expected["dataset_path"] = None
        if _normalized(manifest["bench_spec"]) != expected:
            raise StagingError("recorded bench_spec does not match the private spec")
        attach_references(
            public_path,
            public_bytes,
            manifest,
            bench_spec=_normalized(seeded.to_dict()),
            dataset=dict(
                manifest["dataset"],
                path=(str(Path(spec.dataset_path).resolve()) if spec.dataset_path else None),
                path_redacted=False,
            ),
            qualification=qualification,
            records=len(artifact.records),
        )
        staged.append(public_path.parent)
    return staged
