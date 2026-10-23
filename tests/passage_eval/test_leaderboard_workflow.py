"""Regression checks for the published leaderboard and recording notebook."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from passage_eval import Submission, load_leaderboard
from passage_eval.baselines.classical import GreedyToGoalPlanner
from passage_eval.protocol.planner import evaluate_planner


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def renderer():
    spec = importlib.util.spec_from_file_location(
        "passage_leaderboard_renderer", ROOT / "scripts/passage-eval/render_plotly_leaderboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate_rows():
    rows = []
    for method, regrets in (
        ("reg-repvit_m1_bellman_vbridge", [0.01, 0.09, 0.11]),
        ("seg-convnextv2_atto_baseline", [0.05, 0.05, 0.05]),
    ):
        for seed, regret in zip((1234, 2026, 3407), regrets):
            record = Submission(
                method_name=method,
                family="dense_regression" if method.startswith("reg") else "dense_segmentation",
                architecture="test-backbone",
                seed=seed,
                n_samples=32,
                host="test-host",
                user="not-recorded",
                edge_accelerator="H100",
                split="test",
                cost_weight=62.4,
                metrics={"cost_regret_median": regret},
                latency_e2e_ms_p50=5.0,
                dataset_version="passage@revision-A",
                package_version="1.0.0",
                commit_sha="code-A",
                extra={
                    "metric_contract": "neurips-2026-submission-v1",
                    "task_variant": "obstacles",
                    "dataset_repo": "thalesgroup/passage",
                    "dataset_revision": "revision-A",
                },
            )
            row = record.to_row()
            row.update(
                host_accelerator_slug="h100",
                shortname="reg-RepViT" if method.startswith("reg") else "seg-CNXTv2",
                color="#666666",
                is_oracle=False,
            )
            rows.append(row)
    return pd.DataFrame(rows)


class _CandidateTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td":
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None:
            self._row.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None


def _render_partial(renderer, candidate_rows, tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "REPO_ROOT", tmp_path)
    path = tmp_path / "leaderboard.md"
    renderer.render_partial(candidate_rows, ["h100"], path, iframe_prefix="gen/plotly")
    html = path.read_text()
    table = _CandidateTable()
    table.feed(html)
    return html, table.rows


def test_candidate_table_uses_all_distinct_seeds_instead_of_the_best_seed(
    renderer, candidate_rows, tmp_path, monkeypatch
):
    _, rows = _render_partial(renderer, candidate_rows, tmp_path, monkeypatch)
    scores = {row[1]: float(row[-1].split()[0]) for row in rows}
    assert scores["reg-repvit_m1_bellman_vbridge"] == pytest.approx(0.07)
    assert scores["seg-convnextv2_atto_baseline"] == pytest.approx(0.05)
    assert scores["reg-repvit_m1_bellman_vbridge"] > scores["seg-convnextv2_atto_baseline"]


def _different_revision(candidate_rows):
    alternate = candidate_rows.iloc[[0]].copy()
    alternate["dataset_version"] = "passage@revision-B"
    alternate["cost_regret_median"] = 0.8
    extra = json.loads(alternate.iloc[0].extra_json)
    extra["dataset_revision"] = "revision-B"
    alternate["extra_json"] = json.dumps(extra)
    return pd.concat([candidate_rows, alternate], ignore_index=True)


def test_plotly_scatter_keeps_dataset_revisions_separate(renderer, candidate_rows):
    figure = renderer._per_accelerator_fig(_different_revision(candidate_rows), "h100")
    assert sorted(float(trace.y[0]) for trace in figure.data) == pytest.approx([0.05, 0.07, 0.8])
    hover = " ".join(trace.hovertemplate for trace in figure.data)
    assert "revision-A" in hover and "revision-B" in hover


def test_full_plotly_chart_keeps_dataset_revisions_separate(renderer, candidate_rows, tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "REPO_ROOT", tmp_path)
    output = tmp_path / "leaderboard.html"
    renderer.render_full(_different_revision(candidate_rows), output)
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', output.read_text())
    data, _ = json.JSONDecoder().raw_decode(output.read_text()[match.end() :])
    values = []
    for trace in data:
        column = trace["y"]
        if isinstance(column, dict):
            column = np.frombuffer(base64.b64decode(column["bdata"]), dtype=column["dtype"])
        values.extend(column)
    assert sorted(values) == pytest.approx([0.05, 0.07, 0.8])


def test_candidate_table_discloses_each_condition_and_metric_contract(renderer, candidate_rows, tmp_path, monkeypatch):
    html, rows = _render_partial(renderer, _different_revision(candidate_rows), tmp_path, monkeypatch)
    assert len(rows) == 3
    assert "revision-A" in html and "revision-B" in html
    assert "neurips-2026-submission-v1" in html
    assert 'href="benchmark/protocol/"' in html


def test_plotly_rejects_repeated_seeds_in_a_condition(renderer, candidate_rows):
    repeated = pd.concat([candidate_rows, candidate_rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="(?i)(duplicate|repeat).*seed|seed.*(duplicate|repeat)"):
        renderer._per_accelerator_fig(repeated, "h100")


@pytest.fixture(params=[([0.0, 1.0, np.nan], "2/3"), ([np.nan, np.nan, np.nan], "0/3")])
def incomplete_metric_rows(request, candidate_rows):
    values, coverage = request.param
    rows = candidate_rows.iloc[:3].copy()
    rows["cost_regret_median"] = values
    return rows, coverage


def test_seed_coverage_is_visible_in_candidate_table(renderer, incomplete_metric_rows, tmp_path, monkeypatch):
    frame, coverage = incomplete_metric_rows
    _, rows = _render_partial(renderer, frame, tmp_path, monkeypatch)
    assert coverage in rows[0]
    if coverage == "2/3":
        assert rows[0][-1] == "0.5000 ± 0.7071"
    else:
        assert rows[0][-1].lower() == "unavailable"


def test_seed_coverage_is_visible_in_scatter(renderer, incomplete_metric_rows):
    frame, coverage = incomplete_metric_rows
    figure = renderer._per_accelerator_fig(frame, "h100")
    if coverage == "2/3":
        assert [float(trace.y[0]) for trace in figure.data] == [0.5]
        assert "metric seeds=2/3" in figure.data[0].hovertemplate
        assert "0.5000 ± 0.7071" in figure.data[0].hovertemplate
    else:
        assert not figure.data
        notes = " ".join(annotation.text for annotation in figure.layout.annotations)
        assert "unavailable" in notes.lower() and "metric seeds=0/3" in notes
        assert all(np.isfinite(value) for value in figure.layout.yaxis.range)


def test_seed_coverage_is_visible_in_full_chart(renderer, incomplete_metric_rows, tmp_path, monkeypatch):
    frame, coverage = incomplete_metric_rows
    monkeypatch.setattr(renderer, "REPO_ROOT", tmp_path)
    output = tmp_path / "leaderboard.html"
    renderer.render_full(frame, output)
    document = output.read_text()
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', document)
    decoder = json.JSONDecoder()
    data, consumed = decoder.raw_decode(document[match.end() :])
    layout, _ = decoder.raw_decode(document[match.end() + consumed :].lstrip().removeprefix(",").lstrip())
    if coverage == "2/3":
        assert len(data) == 1 and data[0]["y"] == [0.5]
        assert "metric seeds=2/3" in data[0]["hovertemplate"]
        assert "0.5000 ± 0.7071" in data[0]["hovertemplate"]
    else:
        assert not data
        notes = " ".join(annotation["text"] for annotation in layout["annotations"])
        assert "unavailable" in notes.lower() and "metric seeds=0/3" in notes


def _record_notebook_run(arrays, metrics, task_list):
    from IPython.display import Markdown

    namespace = {
        "Path": Path,
        "np": np,
        "project_root": ROOT,
        "runs": {"Greedy": (SimpleNamespace(name="test-greedy"), metrics, arrays)},
        "resolution": 256,
        "objective": "elevation",
        "condition": "obstacles",
        "split": "validation",
        "dataset_repo": "thalesgroup/passage",
        "dataset_revision": "exact-fixture-revision",
        "dataset_path": None,
        "task_list": task_list,
        "Markdown": Markdown,
        "display": lambda value: None,
    }
    notebook = json.loads((ROOT / "notebooks/passage-eval/03_leaderboard.ipynb").read_text())
    try:
        exec(compile("".join(notebook["cells"][6]["source"]), "03_leaderboard:record", "exec"), namespace)
    except BaseException:
        if "results_root" in namespace:
            shutil.rmtree(namespace["results_root"])
        raise
    return namespace


def test_recording_notebook_preserves_every_scored_array(metre_scene):
    metrics, arrays = evaluate_planner(GreedyToGoalPlanner(), [metre_scene])
    namespace = _record_notebook_run(arrays, metrics, [metre_scene])
    try:
        board = load_leaderboard(namespace["results_root"]).df
        with np.load(namespace["results_root"] / board.iloc[0].per_sample_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    finally:
        shutil.rmtree(namespace["results_root"])


def test_recording_notebook_uses_exported_contract_and_exact_provenance(metre_scene):
    metrics, arrays = evaluate_planner(GreedyToGoalPlanner(), [metre_scene])
    namespace = _record_notebook_run(arrays, metrics, [metre_scene])
    try:
        board = load_leaderboard(namespace["results_root"]).df
        extra = json.loads(board.iloc[0].extra_json)
        assert extra["metric_contract"] == "neurips-2026-submission-v1"
        assert extra["task_variant"] == "obstacles"
        assert extra["dataset_revision"] == "exact-fixture-revision"
        assert extra["sample_ids"] == ["released-0"]
        assert board.iloc[0].dataset_version == "thalesgroup/passage@exact-fixture-revision"
    finally:
        shutil.rmtree(namespace["results_root"])


@pytest.mark.parametrize(
    "contract", [None, np.array(["neurips-2026-submission-v1"]), np.array(""), np.array("legacy-v0")]
)
def test_recording_notebook_rejects_missing_or_invalid_export_contract(metre_scene, contract):
    metrics, arrays = evaluate_planner(GreedyToGoalPlanner(), [metre_scene])
    if contract is None:
        del arrays["metric_contract"]
    else:
        arrays["metric_contract"] = contract
    with pytest.raises(ValueError, match="(?i)metric.contract"):
        _record_notebook_run(arrays, metrics, [metre_scene])
