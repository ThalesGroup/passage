"""Existing human authorization and operator binding; synthetic fixtures only."""
import copy
import importlib.util
from pathlib import Path

import pytest

from passage_eval.protocol import admission as A
spec = importlib.util.spec_from_file_location("authorization_fixture", Path(__file__).with_name("campaign_world.py"))
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def binding(record):
    authorization = {
        'statement': 'The stated plan and S preparation are authorized.',
        'source': 'synthetic conversation reference',
        'approver': None,
        'approved_at': None,
        'scope_sha256': A.authorization_scope_sha256(record),
    }
    return {
        'authorization': authorization,
        'operator_receipt': {'operator': 'fixture operator', 'created_at': '2026-01-01T00:00:00Z',
                             'record_core_sha256': A.record_core_sha256(record),
                             'authorization_sha256': A.pinned.canonical_sha256(authorization)},
    }


def test_existing_authorization_keeps_operator_time_distinct_and_checks_scope(tmp_path):
    world = W.copy_green(tmp_path / 'world')
    record = copy.deepcopy(world.record)
    record['approval'] = binding(record)
    W.save_record(world, record, reapprove=False)
    assert A.check_admission(world.record_path, world.evidence).gate('G1').status == 'GREEN'
    record['protocol']['decisions']['budget']['value']['wall_time_ms'] += 1
    record['approval']['operator_receipt']['record_core_sha256'] = A.record_core_sha256(record)
    W.save_record(world, record, reapprove=False)
    assert 'authorization_scope_mismatch' in [x['code'] for x in A.check_admission(world.record_path, world.evidence).gate('G1').reasons]


def test_existing_authorization_allows_population_and_generated_evidence_updates(tmp_path):
    world = W.copy_green(tmp_path / 'world')
    record = copy.deepcopy(world.record)
    expected = A.authorization_scope_sha256(record)
    record['protocol']['decisions']['population']['value'] = {'n_rows': 1024, 'ids_sha256': 'a' * 64}
    record['evidence']['references']['source_roster'] = None
    record['evidence']['warmup']['selection'] = None
    record['evidence']['edge']['orin'] = None
    record['evidence']['native_chain_rehearsal'] = None
    assert A.authorization_scope_sha256(record) == expected
    record['protocol']['decisions']['roster_selection']['value']['salt'] = 'another'
    assert A.authorization_scope_sha256(record) != expected


def test_operator_preparation_receipt_names_current_record_and_authorization(tmp_path):
    import hashlib
    import json
    import datetime as dt
    world = W.preparation_world(tmp_path / 'preparation')
    record = copy.deepcopy(world.record)
    record['approval'] = binding(record)
    W.save_record(world, record, reapprove=False)
    now = dt.datetime.now(dt.timezone.utc)
    receipt = {
        'schema': 'passage-test-preparation-operator-receipt/v1',
        'scope': 'preparation', 'readers': ['build_roster'],
        'admission_record_sha256': hashlib.sha256(world.record_path.read_bytes()).hexdigest(),
        'authorization_sha256': A.pinned.canonical_sha256(record['approval']['authorization']),
        'operator': 'fixture operator', 'created_at': now.isoformat(),
        'expires_at': (now + dt.timedelta(hours=1)).isoformat(),
    }
    world.approval_path.write_text(json.dumps(receipt))
    grant = W.preparation_grant(world)
    assert grant.readers == ('build_roster',)
    receipt['authorization_sha256'] = '0' * 64
    world.approval_path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match='authorization'):
        W.preparation_grant(world)


def test_operator_receipt_never_permits_changed_scope_before_roster_read(tmp_path):
    import hashlib
    import json
    import datetime as dt
    world = W.preparation_world(tmp_path / 'changed')
    record = copy.deepcopy(world.record)
    record['approval'] = binding(record)
    record['protocol']['decisions']['roster_selection']['value']['salt'] = 'other'
    record['approval']['operator_receipt']['record_core_sha256'] = A.record_core_sha256(record)
    W.save_record(world, record, reapprove=False)
    now = dt.datetime.now(dt.timezone.utc)
    world.approval_path.write_text(json.dumps({
        'schema': 'passage-test-preparation-operator-receipt/v1', 'scope': 'preparation',
        'readers': ['build_roster'], 'admission_record_sha256': hashlib.sha256(world.record_path.read_bytes()).hexdigest(),
        'authorization_sha256': A.pinned.canonical_sha256(record['approval']['authorization']),
        'operator': 'fixture operator', 'created_at': now.isoformat(),
        'expires_at': (now + dt.timedelta(hours=1)).isoformat(),
    }))
    with pytest.raises(PermissionError, match='authorization_scope_mismatch'):
        W.preparation_grant(world)


def test_native_g5_accepts_existing_warmup_provenance_and_checks_actual_roster(tmp_path):
    import datetime as dt
    import json
    from passage_eval.io import roster as roster_io
    world = W.copy_green(tmp_path / 'native')
    record = copy.deepcopy(world.record)
    record['protocol']['profile'] = 'descriptive_native_2026'
    record['protocol']['decisions']['timing_profile']['value'] = 'native_raw_path_persistent_v1'
    warm = json.loads((world.evidence / 'warmup-roster.json').read_bytes())
    warm['selection']['warmup']['evaluated_roster_sha256'] = 'f' * 64
    warm['roster_sha256'] = roster_io.compute_roster_sha256(warm)
    path = world.evidence / 'warmup-roster.json'
    W.write_json(path, warm)
    record['evidence']['warmup']['roster'] = W.ref(world.evidence, path)
    admissible = json.loads((world.evidence / 'admissible-roster.json').read_bytes())
    checker = A._Checker(record, world.evidence, Path('unused'), dt.datetime.now(dt.timezone.utc))
    checker.admitted.update(rehearsal_latency_ok=True, admissible_roster=admissible,
        admissible_roster_sha256=admissible['roster_sha256'],
        admissible_dataset_root_sha256=admissible['dataset_root_identity']['sha256'],
        native_warmup_projections=[A.pinned.canonical_sha256(A.native_input_projection(warm))])
    checker._source_roster = admissible
    assert 'warmup_not_disjoint_from_admissible_roster' not in [r['code'] for r in checker.g5().reasons]
    checker._source_roster = {'tasks': [warm['tasks'][0]]}
    assert 'warmup_not_disjoint_from_admissible_roster' in [r['code'] for r in checker.g5().reasons]
    checker._source_roster = admissible
    checker.admitted['admissible_roster'] = {'tasks': [warm['tasks'][0]]}
    assert 'warmup_not_disjoint_from_admissible_roster' in [r['code'] for r in checker.g5().reasons]


def test_statistics_accepts_both_approval_forms_and_rejects_changed_binding(tmp_path):
    from passage_eval.campaign.statistics import _record_problems
    world = W.copy_green(tmp_path / 'statistics')
    original = copy.deepcopy(world.record)
    aggregate = {'split': original['split'], 'harness': original['harness']}
    seed = original['protocol']['decisions']['seeds']['value'][0]
    assert not any('approval' in p for p in _record_problems(original, seed, aggregate))
    translated = copy.deepcopy(original)
    translated['approval'] = binding(translated)
    assert not any('approval' in p for p in _record_problems(translated, seed, aggregate))
    translated['approval']['authorization']['statement'] = 'Different authorization.'
    assert any('approval' in p for p in _record_problems(translated, seed, aggregate))


def test_preparation_rejects_record_replaced_after_admission_check(tmp_path, monkeypatch):
    import json
    world = W.preparation_world(tmp_path / 'substitution')
    original_check = A.check_admission
    def replace_after_check(*args, **kwargs):
        report = original_check(*args, **kwargs)
        replaced = json.loads(world.record_path.read_bytes())
        replaced['protocol']['decisions']['roster_selection']['value']['salt'] = 'substituted'
        world.record_path.write_text(json.dumps(replaced))
        return report
    monkeypatch.setattr(A, 'check_admission', replace_after_check)
    with pytest.raises(PermissionError):
        W.preparation_grant(world)
