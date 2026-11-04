"""``score-roster``: a roster-aware scorer for supervised runs (the re-review's blocker).

``cmd_score`` reloads the scenes from the spec and cannot reproduce a salted roster, and ``stage-score --roster`` refuses
``--score``. The scorer here runs in its own process, loads the labelled scenes through ``load_roster_scenes``, checks the
run against the pinned roster and calls the unchanged ``score_run_artifact`` / ``write_scored_artifact``. The fixtures are
a tiny pinned Parquet dataset and its roster (see ``test_roster.py``); the dataset root, the labels and the candidate are
synthetic. Nothing here reads the public test split.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from passage_eval import cli
from passage_eval.cli import BenchSpec
from passage_eval.execution import RunArtifact, read_run_artifact, score_run_artifact, write_run_artifact
from passage_eval.io import roster as roster_io
from passage_eval.supervision.__main__ import main as supervision_main
from passage_eval.supervision.roster_scoring import RosterScoringError, score_roster_run

ROSTER_ID = "synthetic-public-test"
# A salt whose first six rows all have a footprint, so the roster plans block analysis (three blocks).
BLOCK_SALT = "synthetic-test-v1/r0003/roster-v1/blocks-0/"
LEADERBOARD = Path(__file__).resolve().parents[3] / "hf-passage-leaderboard-results" / "tools" / "receipt_from_run.py"
CANDIDATE = """
from passage_eval.protocol.contracts import PlanResult


class Planner:
    def __init__(self, **kw):
        pass

    def plan(self, request):
        return PlanResult([request.start, (1, 1), request.goal], work={"expansions": 3}, precision="float32")
"""


def _roster_fixture():
    path = Path(__file__).resolve().parent / "test_roster.py"
    spec = importlib.util.spec_from_file_location("roster_fixture_for_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class World:
    pass


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A pinned synthetic dataset, its 6-of-12 roster and a supervised (process isolation) run of a fixture planner."""
    fixture = _roster_fixture()
    root = tmp_path_factory.mktemp("score-roster")
    w = World()
    w.root, w.fixture = root, fixture
    # A synthetic validation-split snapshot: score-roster refuses a test roster (protocol.guards).
    w.dataset = fixture.write_dataset(root / "dataset", split="validation")
    w.roster, _ = fixture.build(w.dataset, salt=BLOCK_SALT, require_footprints=True, split="validation")
    w.roster_path = root / "roster.json"
    w.roster_path.write_text(json.dumps(w.roster))
    w.sha = w.roster["roster_sha256"]
    w.ids = [task["sample_id"] for task in w.roster["tasks"]]
    candidate = root / "cand"
    (candidate / "rpkg").mkdir(parents=True)
    (candidate / "rpkg" / "__init__.py").write_text("")
    (candidate / "rpkg" / "entry.py").write_text(CANDIDATE)
    w.spec = BenchSpec(
        "score-roster",
        "classical",
        "fixture",
        inference="rpkg.entry:Planner",
        inference_format="request",
        resolution=3,
        split="validation",
        n_samples=6,
        seeds=[1234],
        latency_repeats=2,
        latency_warmup=0,
        budget={"max_expansions": 10, "wall_time_ms": 20000},
        results_root=str(root / "results"),
        dataset_version="pinned synthetic roster dataset",
        reference_policy="independent_analytic_reference",
        reference_cohort_id="synthetic-cohort",
    )
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(w.spec.to_dict()))
    w.run = root / "run"
    assert (
        supervision_main(
            [
                "run",
                "--spec",
                str(spec_path),
                "--output",
                str(w.run),
                "--worker-python",
                sys.executable,
                "--isolation",
                "process",
                "--candidate-path",
                str(candidate),
                "--ipc-baseline-pings",
                "2",
                "--roster",
                str(w.roster_path),
                "--roster-sha256",
                w.sha,
                "--dataset-root",
                str(w.dataset),
                "--roster-id",
                ROSTER_ID,
            ]
        )
        == 0
    )
    return w


def score(world, out, **overrides):
    values = dict(
        run=world.run,
        roster_path=world.roster_path,
        roster_sha256=world.sha,
        dataset_root=world.dataset,
        roster_id=ROSTER_ID,
    )
    values.update(overrides)
    return score_roster_run(
        values["run"],
        values["roster_path"],
        values["roster_sha256"],
        values["dataset_root"],
        out,
        roster_id=values["roster_id"],
    )


@pytest.fixture(scope="module")
def scored(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("scored") / "private"
    return out, score(world, out)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# the scored layout
# --------------------------------------------------------------------------------------------------------------

LAYOUT = {
    "run_manifest.json",
    "requests.jsonl",
    "paths.npz",
    "public_run_manifest.json",
    "per_sample.npz",
    "metrics.json",
    "score_receipt.json",
    "score_reference_inputs.json",
}


def test_a_roster_run_is_scored_into_the_layout_cmd_score_produces(world, scored):
    out, directories = scored
    assert [d.relative_to(out).as_posix() for d in directories] == ["seed-1234/cold", "seed-1234"]  # sorted manifests
    for directory in directories:
        assert LAYOUT <= {p.name for p in directory.iterdir()}
        receipt = json.loads((directory / "score_receipt.json").read_text())
        manifest = json.loads((directory / "run_manifest.json").read_text())
        repeats = manifest["repeats"]
        assert receipt["schema"] == "passage-offline-score-v1" and receipt["qualified_reference"] is True
        assert (
            receipt["metric_contract"] == "passage-qualified-complete-v1"
            and receipt["reference_policy"] == "qualified_exact"
        )
        assert receipt["scored_records"] == 6 * repeats and receipt["scored_repeats"] == repeats
        assert receipt["run_manifest_sha256"] == sha256(directory / "run_manifest.json")
        assert receipt["per_sample_sha256"] == sha256(directory / "per_sample.npz")
        assert receipt["metrics_sha256"] == sha256(directory / "metrics.json")
        assert receipt["reference_inputs_sha256"] == sha256(directory / "score_reference_inputs.json")
        assert (
            receipt["reference_payload_identity"] == manifest["reference_identity"]
        )  # what the leaderboard side checks
        assert receipt["reference_admission_policy"] == "independent_analytic_reference"
        assert receipt["reference_cohort_id"] == "synthetic-cohort"
        with np.load(directory / "per_sample.npz", allow_pickle=False) as arrays:
            assert [str(x) for x in arrays["sample_id"]] == world.ids * repeats  # roster order, every task
            assert arrays["reference_qualified"].all() and np.isfinite(arrays["oracle_costs"]).all()
            assert (arrays["oracle_costs"] > 0).all()
        metrics = json.loads((directory / "metrics.json").read_text())
        assert (
            metrics["n_samples"] == 6
            and metrics["scored_repeat_count"] == repeats
            and metrics["reference_qualified"] is True
        )


def test_the_staged_manifest_differs_from_the_public_one_only_where_the_leaderboard_side_expects(world, scored):
    out, directories = scored
    for directory in directories:
        relative = directory.relative_to(out)
        public_bytes = (world.run / relative / "run_manifest.json").read_bytes()
        public, staged = json.loads(public_bytes), json.loads((directory / "run_manifest.json").read_text())
        assert (directory / "public_run_manifest.json").read_bytes() == public_bytes
        differing = {key for key in set(public) | set(staged) if public.get(key) != staged.get(key)}
        assert differing == {"reference_qualification_withheld", "reference_qualification", "scoring_stage"}
        stage = staged["scoring_stage"]
        assert stage["public_manifest_sha256"] == hashlib.sha256(public_bytes).hexdigest()
        assert (
            stage["schema"] == "passage-scoring-stage-v1" and stage["scorer"] == "passage_eval.supervision score-roster"
        )
        assert stage["roster"] == {"roster_id": ROSTER_ID, "roster_sha256": world.sha, "n_expected": 6}
        assert stage["records"] == 6 * public["repeats"]
        assert set(staged["reference_qualification"]) == set(world.ids)
        assert public["dataset"]["path"] is None and staged["dataset"]["path"] is None  # no dataset location
        assert str(world.dataset) not in json.dumps(staged)


def test_the_public_run_directory_is_not_modified_and_holds_no_reference(world, scored):
    for path in world.run.rglob("*"):
        assert path.name not in {
            "per_sample.npz",
            "metrics.json",
            "score_receipt.json",
            "score_reference_inputs.json",
            "public_run_manifest.json",
        }
    assert "reference_qualification" not in json.loads((world.run / "seed-1234" / "run_manifest.json").read_text())


def test_the_scores_equal_an_independent_computation_with_the_unchanged_scorer(world, scored):
    """Same scenes (``load_roster_scenes``), same references, same ``score_run_artifact``: the files hold exactly that."""
    out, directories = scored
    scenes = roster_io.load_roster_scenes(world.roster, world.dataset)
    qualification = cli._qualification(scenes, world.spec)
    artifact = read_run_artifact(world.run / "seed-1234")
    metrics, arrays = score_run_artifact(
        artifact, scenes, reference_store=qualification, geographic_blocks=roster_io.roster_block_table(world.roster)
    )
    with np.load(out / "seed-1234" / "per_sample.npz", allow_pickle=False) as stored:
        assert set(stored.files) == set(arrays)
        for name in arrays:
            assert np.array_equal(stored[name], arrays[name], equal_nan=stored[name].dtype.kind == "f"), name
    recorded = json.loads((out / "seed-1234" / "metrics.json").read_text())
    for name in ("success_rate", "cost_regret_median", "n_samples", "terminal_ok_rate", "legal_route_rate"):
        assert recorded[name] == pytest.approx(metrics[name]), name
    assert (
        hashlib.sha256(json.dumps(qualification, sort_keys=True).encode()).hexdigest()
        == artifact.manifest["reference_identity"]
    )


def test_the_scores_carry_the_sealed_roster_blocks_and_the_comparison_uses_them(world, scored):
    out, directories = scored
    table = roster_io.roster_block_table(world.roster)
    expected = {task["sample_id"]: task["geographic_block"] for task in world.roster["tasks"]}
    assert table["complete"] and table["blocks"] == expected and len(set(expected.values())) == 3
    for directory in directories:
        with np.load(directory / "per_sample.npz", allow_pickle=False) as arrays:
            assert [str(x) for x in arrays["cluster_id"]] == [expected[str(x)] for x in arrays["sample_id"]]
            assert set(arrays["cluster_provenance"].tolist()) == {"sealed_roster_geographic_block"}
        record = json.loads((directory / "metrics.json").read_text())["geographic_blocks"]
        assert record["roster_sha256"] == world.sha and record["block_table_sha256"] == table["block_table_sha256"]
        assert record["provenance"] == "sealed_roster_geographic_block" and record["number_of_blocks"] == 3
    from passage_eval.metrics.comparison import compare_artifacts

    warm = out / "seed-1234"
    other = out.parent / "copy-of-warm"
    shutil.copytree(warm, other)
    automatic = compare_artifacts([warm, other], bootstrap_resamples=20)
    assert automatic["uncertainty"]["block_provenance"] == "scored_sealed_roster_blocks"
    assert automatic["uncertainty"]["status"] == "declared_blocks"
    assert automatic["comparisons"][0]["all_task"]["cost_regret"]["median"]["n_blocks"] == 3
    explicit = compare_artifacts([warm, other], bootstrap_resamples=20, geographic_blocks=table)
    assert explicit["uncertainty"]["block_provenance"] == "sealed_roster_block_table"
    assert explicit["uncertainty"]["sealed_roster_blocks"]["block_table_sha256"] == table["block_table_sha256"]
    written = out.parent / "comparison.json"
    base = ["compare", "--score", str(warm), "--score", str(other), "--output", str(written)]
    base += ["--bootstrap-resamples", "20"]
    assert cli.main([*base, "--roster", str(world.roster_path), "--roster-sha256", world.sha]) == 0
    assert json.loads(written.read_text())["uncertainty"]["block_provenance"] == "sealed_roster_block_table"
    with pytest.raises(SystemExit):
        cli.main([*base, "--roster", str(world.roster_path)])  # the pin is required with the roster
    with pytest.raises(ValueError, match="expected pin"):
        cli.main([*base, "--roster", str(world.roster_path), "--roster-sha256", "0" * 64])
    shutil.rmtree(other)
    written.unlink()


@pytest.mark.skipif(not LEADERBOARD.is_file(), reason="the leaderboard results repository is not next to this one")
def test_the_leaderboard_receipt_tool_reads_the_scored_layout(world, scored):
    """The consumer of the scored directory: ``receipt_from_run.load_scored_run`` (its own process, standard library only)."""
    out, _ = scored
    code = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('receipt_from_run', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "loaded = module.load_scored_run(sys.argv[2], sys.argv[3])\n"
        "print(json.dumps({'qualified': loaded['qualified'], 'records': len(loaded['records']), "
        "'reference_identity': loaded['manifest']['reference_identity'], 'n': loaded['metrics']['n_samples']}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(LEADERBOARD), str(world.run / "seed-1234"), str(out / "seed-1234")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["qualified"] is True and result["records"] == 12 and result["n"] == 6
    assert (
        result["reference_identity"]
        == json.loads((out / "seed-1234" / "run_manifest.json").read_text())["reference_identity"]
    )


# --------------------------------------------------------------------------------------------------------------
# the command line, in its own process
# --------------------------------------------------------------------------------------------------------------


def test_the_command_runs_in_its_own_process_and_prints_the_scored_directories(world, tmp_path):
    out = tmp_path / "by-process"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "passage_eval.supervision",
            "score-roster",
            "--run",
            str(world.run),
            "--roster",
            str(world.roster_path),
            "--roster-sha256",
            world.sha,
            "--dataset-root",
            str(world.dataset),
            "--output",
            str(out),
            "--roster-id",
            ROSTER_ID,
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [str((out / "seed-1234" / "cold").resolve()), str((out / "seed-1234").resolve())]
    assert (out / "seed-1234" / "score_receipt.json").is_file()
    refused = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "passage_eval.supervision",
            "score-roster",
            "--run",
            str(world.run),
            "--roster",
            str(world.roster_path),
            "--roster-sha256",
            "0" * 64,
            "--dataset-root",
            str(world.dataset),
            "--output",
            str(tmp_path / "never"),
        ],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "roster_sha256 differs" in refused.stderr and not (tmp_path / "never").exists()


def test_the_in_process_entry_point_and_the_required_arguments(world, tmp_path, capsys):
    assert (
        supervision_main(
            [
                "score-roster",
                "--run",
                str(world.run),
                "--roster",
                str(world.roster_path),
                "--roster-sha256",
                world.sha,
                "--dataset-root",
                str(world.dataset),
                "--output",
                str(tmp_path / "o"),
                "--roster-id",
                ROSTER_ID,
            ]
        )
        == 0
    )
    assert len(capsys.readouterr().out.split()) == 2
    for missing in (
        ["--roster-sha256", world.sha],
        ["--dataset-root", str(world.dataset)],
        ["--roster", str(world.roster_path)],
    ):
        arguments = {
            "--run": str(world.run),
            "--roster": str(world.roster_path),
            "--roster-sha256": world.sha,
            "--dataset-root": str(world.dataset),
            "--output": str(tmp_path / "x"),
        }
        del arguments[missing[0]]
        with pytest.raises(SystemExit):
            supervision_main(["score-roster", *[a for item in arguments.items() for a in item]])


def test_the_default_roster_id_is_the_rosters_benchmark_id(world, tmp_path):
    with pytest.raises(
        RosterScoringError, match="records roster id 'synthetic-public-test', expected 'synthetic-test-v1'"
    ):
        score(world, tmp_path / "o", roster_id=None)  # the run was made with --roster-id synthetic-public-test


# --------------------------------------------------------------------------------------------------------------
# fail closed: nothing is written for a run that does not belong to this roster
# --------------------------------------------------------------------------------------------------------------


def edited_copy(world, tmp_path, edit, name="edited"):
    """A copy of the public run whose manifests were rewritten by ``edit`` (the manifest is not covered by the artifact hashes)."""
    run = tmp_path / name
    shutil.copytree(world.run, run)
    for path in run.rglob("run_manifest.json"):
        manifest = json.loads(path.read_text())
        edit(manifest)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return run


def refused(world, tmp_path, run, match, exception=RosterScoringError, **overrides):
    out = tmp_path / "out"
    with pytest.raises(exception, match=match):
        score(world, out, run=run, **overrides)
    assert not out.exists()  # nothing is left behind


def test_a_wrong_roster_pin_is_refused(world, tmp_path):
    refused(
        world, tmp_path, world.run, "roster_sha256 differs from the expected pin", ValueError, roster_sha256="0" * 64
    )


def edits():
    def roster_sha(m):
        m["supervision"]["roster"]["roster_sha256"] = "1" * 64

    def roster_n(m):
        m["supervision"]["roster"]["n_expected"] = 7

    def loaded_n(m):
        m["supervision"]["roster"]["n_loaded"] = 5

    def root_identity(m):
        m["supervision"]["roster"]["dataset_root_identity"]["sha256"] = "2" * 64

    def no_roster(m):
        del m["supervision"]["roster"]

    def unsupervised(m):
        del m["supervision"]

    def boundary(m):
        m["timing_boundary"] = "passage-timing-boundary/v9-unknown"

    def references_present(m):
        m["reference_qualification"] = {}

    def revision(m):
        m["dataset"]["revision"] = "3" * 40

    def resolution(m):
        m["bench_spec"]["resolution"] = 4

    def cost_model(m):
        m["bench_spec"]["cost_model"] = "slope"

    def identity(m):
        m["request_specs"][next(iter(m["request_specs"]))]["elevation_hash"] = "4" * 64

    def objective(m):
        m["request_specs"][next(iter(m["request_specs"]))]["cost_weight"] = 99.0

    def procedure(m):
        m["procedure_id"] = "5" * 64

    return [
        (roster_sha, "different roster"),
        (roster_n, "n_expected=7"),
        (loaded_n, "n_loaded=5"),
        (root_identity, "another dataset root identity"),
        (no_roster, "not a roster run"),
        (unsupervised, "not a supervised public manifest"),
        (boundary, "unknown timing boundary"),
        (references_present, "not a supervised public manifest"),
        (revision, "dataset identity"),
        (resolution, "spec does not match the roster's cell"),
        (cost_model, "spec does not match the roster's cell"),
        (identity, r"ordinary input identity than the roster in \['elevation_hash'\]"),
        (objective, r"ordinary input identity than the roster in \['cost_weight'\]"),
        (procedure, "candidate/procedure identity"),
    ]


@pytest.mark.parametrize("edit, match", edits(), ids=[e.__name__ for e, _ in edits()])
def test_a_run_whose_record_disagrees_with_the_roster_is_refused(world, tmp_path, edit, match):
    refused(world, tmp_path, edited_copy(world, tmp_path, edit), match)


def rewrite(world, tmp_path, name, transform):
    """A run directory rebuilt through ``write_run_artifact`` from the warm series, with ``transform(manifest, records, paths)``."""
    artifact = read_run_artifact(world.run / "seed-1234")
    manifest, records, paths = (
        copy.deepcopy(artifact.manifest),
        copy.deepcopy(artifact.records),
        [p.copy() for p in artifact.paths],
    )
    for record in records:
        for key in ("path_shape", "path_dtype"):
            record.pop(key, None)
    manifest, records, paths = transform(manifest, records, paths)
    manifest.pop("artifact_hashes", None)
    run = tmp_path / name
    write_run_artifact(RunArtifact(manifest, records, paths), run / "seed-1234")
    return run


def test_a_run_that_lacks_a_roster_task_is_refused_whatever_else_is_consistent(world, tmp_path):
    missing = world.ids[-1]

    def drop(manifest, records, paths):
        manifest["roster"] = [sid for sid in manifest["roster"] if sid != missing]
        manifest["request_specs"].pop(missing)
        manifest["request_metadata"].pop(missing)
        keep = [i for i, record in enumerate(records) if record["sample_id"] != missing]
        return manifest, [records[i] for i in keep], [paths[i] for i in keep]

    run = rewrite(world, tmp_path, "missing", drop)
    assert read_run_artifact(run / "seed-1234").records  # a perfectly valid artifact of 5 tasks ...
    refused(
        world,
        tmp_path,
        run,
        r"does not cover every roster task in roster order \(1 roster tasks missing, 0 not in the roster",
    )


def test_a_run_in_another_task_order_or_with_a_foreign_task_is_refused(world, tmp_path):
    def reverse(manifest, records, paths):
        manifest["roster"] = list(reversed(manifest["roster"]))
        return manifest, records, paths

    refused(world, tmp_path, rewrite(world, tmp_path, "reversed", reverse), "in roster order")

    def foreign(manifest, records, paths):
        gone = world.ids[-1]
        manifest["roster"] = [*manifest["roster"][:-1], "foreign-task"]
        manifest["request_specs"]["foreign-task"] = manifest["request_specs"].pop(gone)
        manifest["request_metadata"]["foreign-task"] = manifest["request_metadata"].pop(gone)
        for record in records:
            if record["sample_id"] == gone:
                record["sample_id"] = "foreign-task"
        return manifest, records, paths

    refused(
        world, tmp_path, rewrite(world, tmp_path, "foreign", foreign), r"1 roster tasks missing, 1 not in the roster"
    )


def test_a_dataset_root_that_is_not_the_pinned_one_is_refused(world, tmp_path):
    refused(world, tmp_path, world.run, "no readable PINNED.json", ValueError, dataset_root=tmp_path)
    tampered = tmp_path / "tampered-dataset"
    shutil.copytree(world.dataset, tampered)
    shard = tampered / world.roster["tasks"][0]["source_file"]  # a shard the roster really reads
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF  # same size, one flipped byte
    shard.write_bytes(bytes(data))
    refused(world, tmp_path, world.run, "sha256 differs", ValueError, dataset_root=tampered)
    receipt = tmp_path / "other-receipt"
    shutil.copytree(world.dataset, receipt)
    pinned = json.loads((receipt / "PINNED.json").read_text())
    pinned["tag"] = "another-tag"
    (receipt / "PINNED.json").write_text(json.dumps(pinned, sort_keys=True))
    refused(world, tmp_path, world.run, "receipt sha256 differs", ValueError, dataset_root=receipt)


def test_references_that_differ_from_the_recorded_identity_are_refused(world, tmp_path):
    def other_identity(manifest):
        manifest["reference_identity"] = "6" * 64

    refused(world, tmp_path, edited_copy(world, tmp_path, other_identity), "re-qualified references differ")

    def other_content(manifest):
        manifest["dataset"]["content_hash"] = "7" * 64

    refused(world, tmp_path, edited_copy(world, tmp_path, other_content, "edited-content"), "dataset content differs")


def test_an_existing_output_and_nested_directories_are_refused(world, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        score(world, existing)
    with pytest.raises(RosterScoringError, match="must not contain one another"):
        score(world, world.run / "inside")
    with pytest.raises(RosterScoringError, match="no run artifacts"):
        score_roster_run(tmp_path / "empty-or-missing", world.roster_path, world.sha, world.dataset, tmp_path / "o")


def test_a_failure_while_scoring_removes_the_half_written_directory(world, tmp_path, monkeypatch):
    import passage_eval.execution as execution

    def boom(*args, **kwargs):
        raise RuntimeError("scoring failed midway")

    monkeypatch.setattr(execution, "score_run_artifact", boom)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="midway"):
        score(world, out)
    assert not out.exists()


def test_stage_score_refuses_score_for_a_roster_run_and_points_to_the_roster_scorer(world, tmp_path, capsys):
    spec_path = world.root / "spec.json"
    with pytest.raises(SystemExit):
        supervision_main(
            [
                "stage-score",
                "--run",
                str(world.run),
                "--spec",
                str(spec_path),
                "--output",
                str(tmp_path / "x"),
                "--roster",
                str(world.roster_path),
                "--roster-sha256",
                world.sha,
                "--dataset-root",
                str(world.dataset),
                "--score",
            ]
        )
    assert "score-roster" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------------------
# the test split
# --------------------------------------------------------------------------------------------------------------


def test_a_test_roster_is_refused_before_any_run_or_dataset_file_is_read(world, tmp_path, monkeypatch):
    from passage_eval.io import roster as R
    from passage_eval.protocol.guards import TestSplitRefused

    synthetic_test = world.fixture.write_dataset(tmp_path / "synthetic-test")  # a synthetic split named test
    roster, _ = world.fixture.build(synthetic_test, salt=BLOCK_SALT, require_footprints=True)
    path = tmp_path / "test-roster.json"
    path.write_text(json.dumps(roster))

    def never(*args, **kwargs):
        raise AssertionError("a dataset row was read")

    monkeypatch.setattr(R, "load_roster_scenes", never)
    with pytest.raises(TestSplitRefused, match="score-roster refuses the test split"):
        score_roster_run(world.run, path, roster["roster_sha256"], synthetic_test, tmp_path / "o")
    arguments = ["score-roster", "--run", str(world.run), "--roster", str(path), "--roster-sha256"]
    arguments += [roster["roster_sha256"], "--dataset-root", str(synthetic_test), "--output", str(tmp_path / "o")]
    with pytest.raises(TestSplitRefused, match="score-roster refuses the test split"):
        supervision_main(arguments)
    record = tmp_path / "admission.json"
    record.write_text("{}")
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        supervision_main([*arguments, "--allow-test-split", "--test-admission-record", str(record)])
    # An all-GREEN admission does not admit a roster other than its sealed admissible roster.
    with pytest.raises(TestSplitRefused, match="not the admission's sealed admissible roster"):
        score_roster_run(
            world.run,
            path,
            roster["roster_sha256"],
            synthetic_test,
            tmp_path / "o",
            allow_test=True,
            admission=world.fixture.green_admission(),
        )
    with pytest.raises(TestSplitRefused, match="only together with --allow-test-split"):
        supervision_main([*arguments, "--test-admission-record", str(record)])
    assert not (tmp_path / "o").exists()


def test_score_roster_binds_the_run_to_an_explicit_reference_inventory(world, tmp_path):
    """Judge R1: the re-qualified reference_id of every task is the sealed inventory's (one limit, one policy)."""
    from functools import partial

    from passage_eval.protocol import reference_admission as ra
    from passage_eval.supervision.roster_scoring import RosterScoringError

    def inventory(limit):
        policy = ra.admission_policy(
            world.roster,
            reference_policy="independent_analytic_reference",
            reference_cohort_id="synthetic-cohort",
            reference_max_expansions=limit,
            exclusion_caps=ra.PROPOSED_EXCLUSION_CAPS,
        )
        return ra.build_reference_inventory(
            world.roster, partial(ra.load_admission_scenes, dataset_path=world.dataset), policy
        )

    args = (world.run, world.roster_path, world.sha, world.dataset)
    same = inventory(world.spec.reference_max_expansions)
    assert score_roster_run(*args, tmp_path / "same", roster_id=ROSTER_ID, reference_inventory=same)
    with pytest.raises(RosterScoringError, match="score-roster: 0 task.*6 reference_id"):
        score_roster_run(*args, tmp_path / "other", roster_id=ROSTER_ID, reference_inventory=inventory(9))
    assert not (tmp_path / "other").exists()
