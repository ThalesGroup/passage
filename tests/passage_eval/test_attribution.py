"""Component controls identify learned contribution without fallback relabeling."""
import pytest
import hashlib
import json

from passage_eval.cli import BenchSpec, _validate_spec
from passage_eval.metrics.comparison import compare_artifacts


def test_exact_fallback_cannot_claim_primary_learned_row():
    spec = BenchSpec('hybrid', 'dense_regression', 'model', component_control={
        'variant': 'exact_fallback', 'guide': 'learned', 'checkpoint_id': 'frozen-weight',
        'machinery_id': 'decoder-v1', 'primary_learned': True})
    with pytest.raises(ValueError, match='fallback.*primary'):
        _validate_spec(spec)


def test_prediction_free_control_cannot_declare_learned_guide():
    spec = BenchSpec('control', 'dense_regression', 'model', component_control={
        'variant': 'prediction_free', 'guide': 'learned', 'machinery_id': 'decoder-v1'})
    with pytest.raises(ValueError, match='prediction_free'):
        _validate_spec(spec)


def test_primary_learned_procedure_cannot_hide_actual_exact_fallback(metre_scene):
    from passage_eval.execution import run_requests
    from passage_eval.protocol.contracts import PlanRequest, PlanResult
    class Hybrid:
        def plan(self, request):
            return PlanResult(metre_scene.oracle_path, stage='fallback', stage_work={'fallback': {'expansions': 4}})
    artifact = run_requests(Hybrid(), [PlanRequest.from_scene(metre_scene)], manifest={
        'component_control': {'variant': 'bounded_repair', 'guide': 'learned',
                              'checkpoint_id': 'weights', 'machinery_id': 'decoder'}})
    assert artifact.records[0]['status'] == 'runtime_error'
    assert artifact.records[0]['fallback']
    assert artifact.records[0]['work']['expansions'] == 4


@pytest.mark.parametrize('identity,reported_work,expected', [
    ('different', True, 'checkpoint_identity_mismatch'),
    ('unknown', True, 'checkpoint_identity_unknown'),
    ('same', False, 'incomplete_work_evidence'),
    ('same', True, 'complete_control_matrix'),
])
def test_declared_checkpoint_alias_requires_measured_identity_and_work(identity, reported_work, expected):
    from types import SimpleNamespace
    import numpy as np
    from passage_eval.metrics.comparison import _component_attribution
    variants = ['direct', 'bounded_repair', 'exact_fallback', 'prediction_free', 'exact_classical']
    runs, primary = [], []
    for index, variant in enumerate(variants):
        learned = index < 3
        control = {'variant': variant, 'guide': 'learned' if learned else 'prediction_free' if index == 3 else 'none'}
        if learned:
            control['checkpoint_id'] = 'declared-alias'
        if variant in {'bounded_repair', 'exact_fallback', 'prediction_free'}:
            control['machinery_id'] = 'same-repair'
        records = [{'repeat_index': 0, 'work': {'expansions': 1} if reported_work else {}}]
        hashes = {} if identity == 'unknown' else {'checkpoint': ('a' if identity == 'same' else str(index)) * 64}
        runs.append({'manifest': {'component_control': control, 'checkpoint_hashes': hashes,
                                  'request_specs': {'task': {'budget': {}}}},
                     'artifact': SimpleNamespace(records=records)})
        primary.append({'repeat_index': np.array([0]), 'valid_path': np.array([True]),
                        'regret': np.array([0.]), 'fallback': np.array([index == 2]), 'latency_ms': np.array([1.])})
    result = _component_attribution(runs, primary, variants, ['task'])
    assert result['status'] == expected
    assert result['complete_checkpoint_controls'] == (['declared-alias'] if expected == 'complete_control_matrix' else [])


def test_component_table_uses_actual_work_fallback_and_failure_denominators(metre_scene, tmp_path):
    from passage_eval.execution import run_requests, score_run_artifact, write_scored_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult
    from passage.audit import qualified_reference_for_scene
    class Candidate:
        def plan(self, request):
            return PlanResult(metre_scene.oracle_path, status='timeout', stage='fallback',
                              stage_work={'primary': {'expansions': 2}, 'fallback': {'expansions': 3}})
    references = {metre_scene.sample_id: qualified_reference_for_scene(metre_scene)}
    for name in ('a', 'b'):
        artifact = run_requests(Candidate(), [PlanRequest.from_scene(metre_scene)], output_dir=tmp_path / name,
                               manifest={'data_identity': {'fixture': 'v1'},
                                         'reference_identity': hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest(),
                                         'component_control': {'variant': 'exact_fallback', 'guide': 'learned',
                                                               'checkpoint_id': 'weights', 'machinery_id': 'decode'}})
        metrics, arrays = score_run_artifact(artifact, [metre_scene],
                                           reference_store=references)
        write_scored_artifact(artifact, metrics, arrays)
    comparison = compare_artifacts([tmp_path / 'a', tmp_path / 'b'], bootstrap_resamples=10)
    row = comparison['component_attribution']['rows'][0]
    assert row['denominator'] == 1
    assert row['legal_route_rate'] == 0
    assert row['fallback_rate'] == 1
    assert row['work']['expansions']['mean'] == 5
    assert row['all_task_cost_regret']['mean'] == 999999
    assert comparison['component_attribution']['status'] == 'incomplete_control_matrix'
