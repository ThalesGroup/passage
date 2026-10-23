"""Matched geographic contrasts preserve the declared estimand and population."""

import numpy as np
import pytest

from passage_eval.metrics.comparison import paired_cluster_bootstrap


def test_bootstrap_quantiles_are_difference_of_statistics_not_quantile_of_differences():
    a = np.array([0.0, 10.0, 11.0])
    b = np.array([0.0, 1.0, 20.0])
    result = paired_cluster_bootstrap(a, b, ["west", "east", "north"], statistic="median", resamples=80, seed=7)
    assert result["estimate"] == 9.0
    assert result["estimate"] != np.median(a - b)
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(80):
        idx = rng.integers(0, 3, size=3)
        expected.append(np.median(a[idx]) - np.median(b[idx]))
    np.testing.assert_allclose(result["interval"], np.quantile(expected, [0.025, 0.975]))


def test_bootstrap_resamples_whole_geographic_blocks_instead_of_individual_rows():
    a = np.array([0.0, 0.0, 100.0])
    b = np.zeros(3)
    result = paired_cluster_bootstrap(a, b, ["tile1", "tile1", "tile2"], statistic="p90", resamples=40, seed=9)
    rng = np.random.default_rng(9)
    expected = []
    groups = [np.array([0, 1]), np.array([2])]
    for _ in range(40):
        idx = np.concatenate([groups[i] for i in rng.integers(0, 2, size=2)])
        expected.append(np.quantile(a[idx], 0.9) - np.quantile(b[idx], 0.9))
    assert result["n_blocks"] == 2
    np.testing.assert_allclose(result["interval"], np.quantile(expected, [0.025, 0.975]))


def test_missing_geographic_metadata_does_not_silently_use_iid_bootstrap():
    result = paired_cluster_bootstrap([2, 3], [1, 1], None, resamples=30)
    assert result["estimate"] == 1.5
    assert result["status"] == "unknown_geographic_blocks"
    assert result["interval"] is None


def test_empty_and_single_block_uncertainty_are_explicit():
    assert paired_cluster_bootstrap([], [], [])["status"] == "empty"
    assert paired_cluster_bootstrap([2], [1], ["tile"])["status"] == "insufficient_blocks"


@pytest.mark.parametrize(
    "a,b,blocks", [([1], [1, 2], ["tile"]), ([float("nan")], [1], ["tile"]), ([1, 2], [1, 2], ["tile"])]
)
def test_invalid_bootstrap_arrays_fail(a, b, blocks):
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(a, b, blocks)


def _scored_fixture(tmp_path, scene, *, status="ok", metadata=None, data_identity=None, repeats=1, series="warm"):
    from passage_eval.execution import run_requests, score_run_artifact, write_scored_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult
    from dataclasses import replace

    scenes = scene if isinstance(scene, list) else [scene]
    scenes = [replace(item, metadata=metadata or item.metadata) for item in scenes]
    requests = [PlanRequest.from_scene(item) for item in scenes]
    scene_map = {item.sample_id: item for item in scenes}
    from passage.audit import qualified_reference_for_scene
    import hashlib
    import json

    references = {item.sample_id: qualified_reference_for_scene(item) for item in scenes}
    assert all(reference["qualified"] for reference in references.values())

    class Direct:
        calls = 0

        def plan(self, request):
            selected = status[self.calls] if isinstance(status, list) else status
            self.calls += 1
            return PlanResult(scene_map[request.sample_id].oracle_path, status=selected)

    manifest = {
        "data_identity": data_identity or {"source": "fixture", "revision": "frozen", "split": "test"},
        "reference_identity": hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest(),
        "seed": 1234,
    }
    run = run_requests(Direct(), requests, output_dir=tmp_path, manifest=manifest, repeats=repeats, series=series)
    metrics, arrays = score_run_artifact(run, scenes, reference_store=references)
    write_scored_artifact(run, metrics, arrays)
    return tmp_path


def test_comparison_empty_both_valid_geometry_and_all_failure_penalty_are_explicit(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, status="timeout")
    b = _scored_fixture(tmp_path / "b", metre_scene)
    result = compare_artifacts([a, b], bootstrap_resamples=20)
    assert result["methods"][0]["all_task"]["cost_regret"]["mean"] == 999999
    assert result["methods"][0]["rates"]["historical_connected_path_rate"] == 1
    assert result["methods"][0]["rates"]["legal_route_rate"] == 0
    assert result["methods"][0]["all_repeat_latency_ms"]["n"] == 1
    paired = result["comparisons"][0]
    assert paired["both_valid_geometry"]["denominator"] == 0
    assert paired["both_valid_geometry"]["assd"]["mean"]["status"] == "empty"
    assert paired["all_task"]["cost_regret"]["mean"]["estimate"] == 999999


def test_actual_source_identity_mismatch_fails_even_when_sample_ids_match(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene, data_identity={"source": "other", "revision": "frozen"})
    with pytest.raises(ValueError, match="source identity"):
        compare_artifacts([a, b])


def test_cold_and_warm_conditions_do_not_enter_the_same_latency_comparison(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, series="cold")
    b = _scored_fixture(tmp_path / "b", metre_scene, series="warm")
    with pytest.raises(ValueError, match="same.*series"):
        compare_artifacts([a, b])


def test_disjoint_geographic_populations_are_never_paired_by_row_order(metre_scene, tmp_path):
    from dataclasses import replace
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", replace(metre_scene, sample_id="other-task"))
    with pytest.raises(ValueError, match="roster"):
        compare_artifacts([a, b])


def test_score_hash_tampering_and_unqualified_reference_fail_closed(metre_scene, tmp_path):
    import json
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["qualified_reference"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="qualified"):
        compare_artifacts([a, b])
    receipt["qualified_reference"] = True
    receipt["per_sample_sha256"] = "changed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash"):
        compare_artifacts([a, b])


@pytest.mark.parametrize(
    "forgery", ["reference_cost", "reference_id", "positive_assd", "positive_h95", "negative_geometry"]
)
def test_rehashed_scores_cannot_forge_qualified_reference_or_geometry(metre_scene, tmp_path, forgery):
    import json
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene)
    with np.load(b / "per_sample.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if forgery == "reference_cost":
        arrays["oracle_costs"] *= 2
        arrays["regret"] = arrays["pred_costs"] / arrays["oracle_costs"] - 1
    elif forgery == "reference_id":
        arrays["reference_id"] = np.full(len(arrays["sample_id"]), "not-the-qualified-reference")
    elif forgery == "positive_assd":
        arrays["assd"][:] = 12
    elif forgery == "positive_h95":
        arrays["h95"][:] = 12
    else:
        arrays["assd"][:] = -123
        arrays["h95"][:] = -456
    np.savez(b / "per_sample.npz", **arrays)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(b / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    if forgery in {"reference_cost", "reference_id"}:
        with np.load(a / "per_sample.npz", allow_pickle=False) as archive:
            first_arrays = {key: archive[key] for key in archive.files}
        for key in ("oracle_costs", "regret") if forgery == "reference_cost" else ("reference_id",):
            first_arrays[key] = arrays[key].copy()
        np.savez(a / "per_sample.npz", **first_arrays)
        first_receipt_path = a / "score_receipt.json"
        first_receipt = json.loads(first_receipt_path.read_text())
        first_receipt["per_sample_sha256"] = sha256_file(a / "per_sample.npz")
        first_receipt_path.write_text(json.dumps(first_receipt))
    with pytest.raises(ValueError, match="reference|geometry"):
        compare_artifacts([a, b])


def test_declared_primary_repeat_is_explicit_and_other_latencies_remain(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, repeats=2)
    b = _scored_fixture(tmp_path / "b", metre_scene, repeats=2)
    result = compare_artifacts([a, b], primary_repeat_index=1, bootstrap_resamples=20)
    assert result["primary_repeat_index"] == 1
    assert result["methods"][0]["all_task"]["denominator"] == 1
    assert result["methods"][0]["all_repeat_latency_ms"]["n"] == 2
    assert result["methods"][0]["seed_condition"]["seed"] == 1234
    assert result["comparisons"][0]["all_task"]["cost_regret"]["median"]["status"] == "unknown_geographic_blocks"


def test_actual_candidate_input_mismatch_fails_with_same_task_id(tensor_scene, tmp_path):
    from dataclasses import replace
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", tensor_scene)
    changed = tensor_scene.input_tensor.copy()
    changed[:, :, 0] = 0.5
    b = _scored_fixture(
        tmp_path / "b", replace(tensor_scene, input_tensor=changed, metadata={"calibration": {"min": -1.0, "max": 1.0}})
    )
    with pytest.raises(ValueError, match="input/task/objective/graph"):
        compare_artifacts([a, b])


def test_duplicate_scored_rows_fail_even_when_score_hash_is_updated(metre_scene, tmp_path):
    import json
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, repeats=2)
    b = _scored_fixture(tmp_path / "b", metre_scene, repeats=2)
    with np.load(b / "per_sample.npz", allow_pickle=False) as store:
        arrays = {key: store[key] for key in store.files}
    arrays["repeat_index"][1] = 0
    arrays["primary_quality"][1] = True
    np.savez(b / "per_sample.npz", **arrays)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(b / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete unique task roster"):
        compare_artifacts([a, b])


def test_failed_secondary_repeat_latency_and_status_are_retained(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, repeats=2, status=["ok", "oom"])
    b = _scored_fixture(tmp_path / "b", metre_scene, repeats=2)
    result = compare_artifacts([a, b], bootstrap_resamples=10)
    method = result["methods"][0]
    assert method["all_task"]["cost_regret"]["mean"] == 0
    assert method["all_repeat_latency_ms"]["n"] == 2
    assert method["terminal_status_counts_all_repeats"] == {"ok": 1, "oom": 1}


def test_explicit_geographic_provenance_is_not_inferred_from_sample_ids(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, metadata={"geographic": {"tile_id": "whole-tile"}})
    b = _scored_fixture(tmp_path / "b", metre_scene, metadata={"geographic": {"tile_id": "whole-tile"}})
    result = compare_artifacts([a, b], bootstrap_resamples=10)
    assert result["uncertainty"]["block_provenance"] == "geographic.tile_id"
    assert result["comparisons"][0]["all_task"]["cost_regret"]["median"]["status"] == "insufficient_blocks"


def test_report_roundtrip_retains_unknown_and_empty_geometry(metre_scene, tmp_path):
    import json
    from passage_eval.metrics.comparison import compare_artifacts, report_artifact

    a = _scored_fixture(tmp_path / "a", metre_scene, status="cancelled")
    b = _scored_fixture(tmp_path / "b", metre_scene)
    result = compare_artifacts([a, b], bootstrap_resamples=10)
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(result, allow_nan=False))
    assert report_artifact(path, format="json") == result
    report = report_artifact(path)
    assert "999,999" in report
    assert "unknown_geographic_blocks" in report
    assert "Both-valid" in report


def test_odd_spec_is_frozen_and_numeric_joint_cells_preserve_input_descriptors(metre_scene, tmp_path):
    import hashlib
    import json
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene)
    spec = {
        "axes": ["elev_std", "endpoint_separation_normalized", "obstacle_fraction"],
        "edges": {"elev_std": [0, 1], "endpoint_separation_normalized": [0, 0.5, 1], "obstacle_fraction": [0, 0.1, 1]},
    }
    result = compare_artifacts([a, b], odd_spec=spec, bootstrap_resamples=10)
    expected_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert result["joint_odd_coverage"]["spec_sha256"] == expected_hash
    assert result["joint_odd_coverage"]["spec"] == spec
    cell = result["joint_odd_coverage"]["cells"][0]
    assert cell["coordinates"] == {"elev_std": 0, "endpoint_separation_normalized": 1, "obstacle_fraction": 0}
    assert result["joint_odd_coverage"]["common_support_status"] == "present"
    assert result["joint_odd_coverage"]["bin_unknown_counts"]["0:a"] == dict.fromkeys(spec["axes"], 0)


def test_absent_odd_strata_are_unknown_despite_matching_task_categories(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene)
    coverage = compare_artifacts([a, b], bootstrap_resamples=10)["joint_odd_coverage"]
    assert coverage["status"] == "unknown_strata"
    assert coverage["common_support_status"] == "unknown"
    assert coverage["unknown_required_strata"] == ["elev_std", "endpoint_separation_normalized", "obstacle_fraction"]


def test_reordered_score_rows_pair_by_task_identity_and_preserve_declared_blocks(metre_scene, tmp_path):
    import json
    from dataclasses import replace
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = [
        replace(metre_scene, metadata={"geographic": {"tile_id": "west"}}),
        replace(metre_scene, sample_id="second", metadata={"geographic": {"tile_id": "east"}}),
    ]
    a = _scored_fixture(tmp_path / "a", scenes, status=["ok", "timeout"])
    b = _scored_fixture(tmp_path / "b", scenes, status=["timeout", "ok"])
    with np.load(b / "per_sample.npz", allow_pickle=False) as store:
        arrays = {key: store[key] for key in store.files}
    arrays = {key: value[::-1] if value.ndim == 1 else value for key, value in arrays.items()}
    np.savez(b / "per_sample.npz", **arrays)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(b / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    result = compare_artifacts([a, b], bootstrap_resamples=40)
    estimate = result["comparisons"][0]["all_task"]["cost_regret"]["mean"]
    assert estimate["estimate"] == 0
    assert estimate["status"] == "estimated"
    assert estimate["n_blocks"] == 2
    assert estimate["interval"][0] < 0 < estimate["interval"][1]
    assert result["comparisons"][0]["both_valid_geometry"]["denominator"] == 0


def test_missing_score_row_is_rejected_after_hash_is_updated(metre_scene, tmp_path):
    import json
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, repeats=2)
    b = _scored_fixture(tmp_path / "b", metre_scene, repeats=2)
    with np.load(b / "per_sample.npz", allow_pickle=False) as store:
        arrays = {key: store[key] for key in store.files}
    arrays = {key: value[:1] if value.ndim == 1 else value for key, value in arrays.items()}
    np.savez(b / "per_sample.npz", **arrays)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(b / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete terminal record"):
        compare_artifacts([a, b])


@pytest.mark.parametrize("contract", ["passage-stored-diagnostic-v1", "unknown-protocol"])
def test_same_diagnostic_or_unknown_contract_is_rejected_despite_qualified_receipts(metre_scene, tmp_path, contract):
    import json
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    paths = [_scored_fixture(tmp_path / name, metre_scene) for name in ("a", "b")]
    for path in paths:
        with np.load(path / "per_sample.npz", allow_pickle=False) as store:
            arrays = {key: store[key] for key in store.files}
        arrays["metric_contract"] = np.array(contract)
        np.savez(path / "per_sample.npz", **arrays)
        receipt_path = path / "score_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["per_sample_sha256"] = sha256_file(path / "per_sample.npz")
        receipt["metric_contract"] = contract
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="supported qualified metric contract"):
        compare_artifacts(paths)


def test_score_receipt_metric_contract_must_equal_array_contract(metre_scene, tmp_path):
    import json
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene)
    b = _scored_fixture(tmp_path / "b", metre_scene)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["metric_contract"] = "different"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt metric contract"):
        compare_artifacts([a, b])


def test_array_primary_repeat_index_must_bind_manifest_and_receipt(metre_scene, tmp_path):
    import json
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    a = _scored_fixture(tmp_path / "a", metre_scene, repeats=2)
    b = _scored_fixture(tmp_path / "b", metre_scene, repeats=2)
    with np.load(b / "per_sample.npz", allow_pickle=False) as store:
        arrays = {key: store[key] for key in store.files}
    arrays["quality_repeat_index"] = np.array(1)
    np.savez(b / "per_sample.npz", **arrays)
    receipt_path = b / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(b / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="primary repeat index"):
        compare_artifacts([a, b])


@pytest.mark.parametrize("cluster_location", ["top_level", "geographic"])
def test_declared_cluster_keeps_distinct_tiles_in_one_bootstrap_block(metre_scene, tmp_path, cluster_location):
    from dataclasses import replace
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = []
    for sample_id, tile_id in [("first", "T1"), ("second", "T2")]:
        metadata = {"geographic": {"tile_id": tile_id}}
        if cluster_location == "top_level":
            metadata["cluster_id"] = "C"
        else:
            metadata["geographic"]["cluster_id"] = "C"
        scenes.append(replace(metre_scene, sample_id=sample_id, metadata=metadata))
    a = _scored_fixture(tmp_path / "a", scenes, status=["ok", "timeout"])
    b = _scored_fixture(tmp_path / "b", scenes, status=["timeout", "ok"])
    with np.load(a / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["cluster_id"].tolist() == ["C", "C"]
    result = compare_artifacts([a, b], bootstrap_resamples=40)
    estimate = result["comparisons"][0]["all_task"]["cost_regret"]["mean"]
    assert estimate["n_blocks"] == 1
    assert estimate["status"] == "insufficient_blocks"
    assert estimate["interval"] is None


def test_partial_declared_cluster_cannot_fall_back_to_complete_tile_metadata(metre_scene, tmp_path):
    from dataclasses import replace
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = [
        replace(metre_scene, sample_id="first", metadata={"cluster_id": "C", "geographic": {"tile_id": "T1"}}),
        replace(metre_scene, sample_id="second", metadata={"geographic": {"tile_id": "T2"}}),
    ]
    a = _scored_fixture(tmp_path / "a", scenes, status=["ok", "timeout"])
    b = _scored_fixture(tmp_path / "b", scenes, status=["timeout", "ok"])
    result = compare_artifacts([a, b], bootstrap_resamples=40)
    estimate = result["comparisons"][0]["all_task"]["cost_regret"]["mean"]
    assert result["uncertainty"]["block_provenance"] == "unknown"
    assert estimate["status"] == "unknown_geographic_blocks"
    assert estimate["interval"] is None


def test_declared_support_cluster_keeps_source_footprints_in_one_component():
    from passage_eval.metrics.comparison import _geographic_blocks

    manifest = {
        "request_metadata": {
            "a": {"cluster_id": "component", "source_footprint_id": "footprint1", "source_tile_id": "T1"},
            "b": {"cluster_id": "component", "source_footprint_id": "footprint2", "source_tile_id": "T2"},
        }
    }
    blocks, _ = _geographic_blocks(manifest, ["a", "b"])
    estimate = paired_cluster_bootstrap([0, 999999], [999999, 0], blocks, statistic="mean", resamples=40)
    assert estimate["n_blocks"] == 1
    assert estimate["status"] == "insufficient_blocks"
    assert estimate["interval"] is None
