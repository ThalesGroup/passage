"""Projected population qualification and a frozen diagnostic audit roster."""
import json
import fcntl
import os
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage.dataset import DatasetSource


def scene(sample_id, split, *, tile='N001E001', relief=250.0, occupancy=0.1, box=(0, 1, 0, 1)):
    west, east, south, north = box
    return {'id': sample_id, 'tensor': b'never-decode', 'image': b'never-read',
            'metadata': {'resolution': 64, 'split': split, 'calibration': {'min': -430., 'max': 8767.},
                         'crop': {'min_elevation': 0., 'max_elevation': relief,
                                  'top_left': {'lat': north, 'lon': west},
                                  'bottom_right': {'lat': south, 'lon': east}},
                         'tile': {'source': {'name': tile, 'lat': 1, 'lon': 1},
                                  'used': [{'name': tile}], 'overlapping': [{'name': tile}]},
                         'obstacles': {'ratio_actual': occupancy},
                         'paths': {'outcome': 42.}}}


def write(root, split, rows, *, name='part-0.parquet'):
    directory = root / '64x64' / split
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / name, row_group_size=2)


def qualify(root, output, **kwargs):
    from passage.qualification import run_metadata_census
    return run_metadata_census(DatasetSource(root, repo='fixture', revision='a' * 40), output,
                               resolutions=(64,), seed=17, batch_size=2, **kwargs)


def test_projected_census_duplicates_missing_and_fixed_bins(tmp_path, monkeypatch):
    original = pq.ParquetFile.iter_batches
    projections = []

    def traced(self, *args, **kwargs):
        projections.append(kwargs.get('columns'))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', traced)
    root, output = tmp_path / 'data', tmp_path / 'out'
    rows = [scene('same', 'train', relief=200, occupancy=.05), scene('same', 'train'),
            scene(None, 'train', relief=-1, occupancy=1.1), scene('missing', 'train')]
    rows[-1]['metadata']['calibration'] = None
    rows[-1]['metadata']['obstacles']['ratio_actual'] = None
    write(root, 'train', rows)
    result = qualify(root, output)
    cell = result['cells']['64/train']
    assert cell['rows'] == 4
    assert cell['duplicate_id_rows'] == 1
    assert cell['missing']['id'] == 1
    assert cell['calibration']['invalid_rows'] == 1
    assert cell['relief']['bins']['[200,500)'] == 3
    assert cell['relief']['out_of_range'] == 1
    assert cell['occupancy']['missing'] == 1
    assert cell['occupancy']['out_of_range'] == 1
    assert result['cells']['64/test']['status'] == 'missing'
    assert all(p and all(name == 'id' or name.startswith('metadata') for name in p) for p in projections)
    assert any('metadata.crop' in p and 'metadata.paths' not in p for p in projections)
    with sqlite3.connect(output / 'census.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM scenes').fetchone()[0] == 4


def test_resume_atomic_shards_and_protocol_identity(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    write(root, 'train', [scene('b', 'train')], name='part-1.parquet')
    calls = []

    def stop(event):
        calls.append(event)
        if event.get('event') == 'shard_complete':
            raise RuntimeError('interruption')

    with pytest.raises(RuntimeError, match='interruption'):
        qualify(root, output, progress=stop)
    assert (output / 'protocol.json').exists()
    first = json.loads((output / 'summary.json').read_text())
    assert first['cells']['64/train']['rows'] == 1
    assert first['status'] == 'failed'
    assert first['failure']['error'] == 'interruption'
    result = qualify(root, output)
    assert result['cells']['64/train']['rows'] == 2
    before = (output / 'audit_roster.json').read_bytes()
    assert qualify(root, output)['status'] == 'complete'
    assert (output / 'audit_roster.json').read_bytes() == before
    from passage.qualification import run_metadata_census
    with pytest.raises(ValueError, match='protocol'):
        run_metadata_census(DatasetSource(root, repo='fixture', revision='a' * 40), output,
                            resolutions=(64,), seed=18)


def test_stratified_roster_deterministic_distinct_tiles_and_locator(tmp_path):
    from passage.qualification import read_roster_row
    root, output = tmp_path / 'data', tmp_path / 'out'
    for split in ('train', 'validation', 'calibration', 'test'):
        rows = [scene(str(i), split, tile=f'T{i % 20:02}', relief=(i % 5) * 300,
                      occupancy=(i % 5) * .1) for i in range(50)]
        write(root, split, rows)
    qualify(root, output)
    roster = json.loads((output / 'audit_roster.json').read_text())
    assert len(roster['scenes']) == 28
    assert roster['population_estimation'] is False
    assert roster['roster_sha256']
    validation = [r for r in roster['scenes'] if r['split'] == 'validation']
    assert len({r['source_tile'] for r in validation}) == 16
    assert len({r['stratum'] for r in validation}) > 1
    for entry in roster['scenes']:
        selected = read_roster_row(DatasetSource(root), entry, columns=['id', 'metadata'])
        assert selected['id'] == entry['sample_id']
        assert selected['metadata'] == entry['metadata']
    other = tmp_path / 'other'
    qualify(root, other)
    assert json.loads((other / 'audit_roster.json').read_text()) == roster


def test_crosssplit_overlap_boundary_and_source_support_separate(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train', box=(0, 1, 0, 1)),
                          scene('b', 'train', box=(10, 11, 10, 11))])
    write(root, 'test', [scene('c', 'test', box=(.5, 1.5, .5, 1.5)),
                         scene('d', 'test', box=(11, 12, 10, 11))])
    result = qualify(root, output)
    geography = result['geography']['64']
    pair = geography['split_pairs']['train/test']
    assert pair['positive_overlap_rows'] == {'train': 1, 'test': 1}
    assert pair['boundary_contact_rows'] == {'train': 1, 'test': 1}
    assert pair['shared_source_tiles'] == 1
    assert geography['source_cell_windows'] == 'unknown'
    assert geography['status'] == 'complete'


def test_source_mutation_is_rejected_after_freeze(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    qualify(root, output)
    write(root, 'train', [scene('a', 'train'), scene('b', 'train')])
    with pytest.raises(ValueError, match='source|protocol'):
        qualify(root, output)


def test_missing_columns_are_reported_without_decoding(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [{'id': 'a', 'tensor': b'x'}])
    result = qualify(root, output)
    assert result['cells']['64/train']['status'] == 'invalid_schema'
    assert result['cells']['64/train']['rows'] == 1
    assert result['cells']['64/train']['missing']['metadata'] == 1
    assert result['roster']['selected'] == 0


def test_outcome_metadata_does_not_change_selection_rank(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    rows = [scene(str(i), 'validation', tile=f'T{i % 5}', relief=i * 100) for i in range(40)]
    write(root, 'validation', rows)
    qualify(root, output, spatial=False)
    first = json.loads((output / 'audit_roster.json').read_text())['scenes']
    for row in rows:
        row['metadata']['paths']['outcome'] = -999.
    write(root, 'validation', rows)
    qualify(root, tmp_path / 'new-out', spatial=False)
    second = json.loads((tmp_path / 'new-out' / 'audit_roster.json').read_text())['scenes']
    assert [(s['sample_id'], s['rank']) for s in first] == [(s['sample_id'], s['rank']) for s in second]


def test_nonfinite_and_wrapping_geography_are_excluded_explicitly(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    rows = [scene('bad', 'train', relief=float('nan'), occupancy=float('inf')),
            scene('wrap', 'train', box=(179, -179, 1, 2))]
    write(root, 'train', rows)
    cell = qualify(root, output)['cells']['64/train']
    assert cell['relief']['invalid'] == 1
    assert cell['occupancy']['invalid'] == 1
    assert cell['footprints']['antimeridian_or_reversed_longitude'] == 1
    roster = json.loads((output / 'audit_roster.json').read_text())
    assert roster['scenes'][0]['metadata']


def test_rtree_rounding_does_not_create_false_overlap_or_contact(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train', box=(95.1, 95.2, 28.1, 28.2))])
    write(root, 'test', [scene('b', 'test', box=(95.2000000001, 95.3, 28.1, 28.2))])
    pair = qualify(root, output)['geography']['64']['split_pairs']['train/test']
    assert pair['positive_overlap_rows'] == {'train': 0, 'test': 0}
    assert pair['boundary_contact_rows'] == {'train': 0, 'test': 0}


def test_crash_after_shard_commit_before_summary_refresh_recovers(tmp_path, monkeypatch):
    import passage.qualification as module
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    original = module._summary
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('between commit and summary')
        return original(*args, **kwargs)

    monkeypatch.setattr(module, '_summary', interrupt)
    with pytest.raises(RuntimeError, match='between commit'):
        qualify(root, output)
    monkeypatch.setattr(module, '_summary', original)
    assert qualify(root, output)['cells']['64/train']['rows'] == 1


def test_roster_tampering_is_rejected(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    qualify(root, output)
    roster = json.loads((output / 'audit_roster.json').read_text())
    roster['scenes'][0]['sample_id'] = 'tampered'
    (output / 'audit_roster.json').write_text(json.dumps(roster))
    with pytest.raises(ValueError, match='roster hash'):
        qualify(root, output)


def test_output_overlap_and_concurrent_writer_are_rejected(tmp_path):
    root = tmp_path / 'data'
    write(root, 'train', [scene('a', 'train')])
    for output in (root, root / 'outputs', tmp_path):
        with pytest.raises(ValueError, match='outside|overlap'):
            qualify(root, output)
    output = tmp_path / 'out'
    output.mkdir()
    with (output / '.qualification.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active writer'):
            qualify(root, output)
    assert not (output / 'protocol.json').exists()


def test_same_size_mtime_replacement_invalidates_resume(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    qualify(root, output)
    shard = root / '64x64' / 'train' / 'part-0.parquet'
    prior = shard.stat()
    replacement = tmp_path / 'replacement.parquet'
    replacement.write_bytes(shard.read_bytes())
    os.utime(replacement, ns=(prior.st_atime_ns, prior.st_mtime_ns))
    replacement.replace(shard)
    assert shard.stat().st_size == prior.st_size
    assert shard.stat().st_mtime_ns == prior.st_mtime_ns
    with pytest.raises(ValueError, match='source|protocol'):
        qualify(root, output)


def test_outside_shard_symlink_is_rejected_before_protocol(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    shard = root / '64x64' / 'train' / 'part-0.parquet'
    outside = tmp_path / 'outside.parquet'
    shard.replace(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match='outside'):
        qualify(root, output)
    assert not (output / 'protocol.json').exists()


def auth_manifest(root, path):
    shards = sorted(root.rglob('*.parquet'))
    value = {'schema_version': 1, 'root': str(root.resolve()), 'repo': 'fixture', 'revision': 'a' * 40,
             'repo_type': 'dataset', 'files': [{'path': str(shard.relative_to(root)),
              'size': shard.stat().st_size, 'algorithm': 'sha256', 'expected_hash': 'b' * 64} for shard in shards]}
    path.write_text(json.dumps(value))
    return value


def test_authentication_manifest_binds_identity_objects_and_rejects_extras(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    manifest_path = tmp_path / 'authentication-manifest.json'
    expected = auth_manifest(root, manifest_path)
    result = qualify(root, output, authentication_manifest=manifest_path)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['authentication_manifest']['manifest_sha256']
    assert result['content_verified'] is False
    record = protocol['manifest']['64/train']['shards'][0]
    assert record['expected_object'] == expected['files'][0]
    write(root, 'train', [scene('extra', 'train')], name='extra.parquet')
    with pytest.raises(ValueError, match='extra|manifest'):
        qualify(root, tmp_path / 'extra-out', authentication_manifest=manifest_path)
    expected['repo'] = 'wrong'
    manifest_path.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match='identity'):
        qualify(root, tmp_path / 'wrong-out', authentication_manifest=manifest_path)


def test_source_changed_after_scan_rejected_before_completion(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    shard = root / '64x64' / 'train' / 'part-0.parquet'

    def changed(event):
        if event['event'] == 'roster_frozen':
            prior = shard.stat()
            os.utime(shard, ns=(prior.st_atime_ns, prior.st_mtime_ns + 1))

    with pytest.raises(ValueError, match='changed|source'):
        qualify(root, output, progress=changed)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] != 'complete'


def test_source_change_during_read_rolls_back_and_reports_shard(tmp_path, monkeypatch):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    shard = root / '64x64' / 'train' / 'part-0.parquet'
    original = pq.ParquetFile.iter_batches
    changed = False

    def mutating(self, *args, **kwargs):
        nonlocal changed
        for batch in original(self, *args, **kwargs):
            if not changed:
                changed = True
                prior = shard.stat()
                os.utime(shard, ns=(prior.st_atime_ns, prior.st_mtime_ns + 1))
            yield batch

    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', mutating)
    with pytest.raises(ValueError, match='changed'):
        qualify(root, output)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'failed'
    assert summary['failure']['shard'] == '64x64/train/part-0.parquet'
    assert summary['cells']['64/train']['rows'] == 0
    with sqlite3.connect(output / 'census.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM scenes').fetchone()[0] == 0


def test_malformed_nested_metadata_is_counted_and_json_safe(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    row = scene('a', 'train')
    row['metadata']['split'] = ['train']
    row['metadata']['provenance'] = {'bad_binary': b'abc'}
    write(root, 'train', [row])
    result = qualify(root, output)
    assert result['cells']['64/train']['invalid']['split'] == 1
    assert result['cells']['64/train']['split_mismatch'] == 1
    roster = json.loads((output / 'audit_roster.json').read_text())
    assert roster['scenes'][0]['metadata']['provenance']['bad_binary'] == {'binary_hex': '616263'}


def test_output_symlinks_cannot_mutate_source(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    output.mkdir()
    shard = root / '64x64' / 'train' / 'part-0.parquet'
    original = shard.read_bytes()
    (output / 'protocol.json.tmp').symlink_to(shard)
    qualify(root, output)
    assert shard.read_bytes() == original
    (output / 'summary.json').unlink()
    (output / 'summary.json').symlink_to(shard)
    with pytest.raises(ValueError, match='symlink'):
        qualify(root, output)
    assert shard.read_bytes() == original


def test_roster_reader_rejects_changed_frozen_shard(tmp_path):
    from passage.qualification import read_roster_row
    root, output = tmp_path / 'data', tmp_path / 'out'
    write(root, 'train', [scene('a', 'train')])
    qualify(root, output)
    entry = json.loads((output / 'audit_roster.json').read_text())['scenes'][0]
    shard = root / entry['locator']['shard']
    prior = shard.stat()
    os.utime(shard, ns=(prior.st_atime_ns, prior.st_mtime_ns + 1))
    with pytest.raises(ValueError, match='changed'):
        read_roster_row(DatasetSource(root), entry, columns=['id', 'metadata'])


def test_incremental_aggregates_and_schema_deduplication_across_shards(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'out'
    first = scene('duplicate', 'train', tile='T1')
    first['metadata']['tile']['used'].append({'name': 'T2'})
    second = scene('duplicate', 'train', tile='T2')
    second['metadata']['tile']['overlapping'].append({'name': 'T1'})
    write(root, 'train', [first])
    write(root, 'train', [second, scene('duplicate', 'train', tile='T2')], name='part-1.parquet')
    result = qualify(root, output)
    cell = result['cells']['64/train']
    assert cell['duplicate_id_rows'] == 2
    assert cell['duplicate_id_values'] == 1
    assert cell['source_tiles'] == {'T1': 1, 'T2': 2}
    assert cell['support_tiles'] == {'T1': 2, 'T2': 3}
    assert cell['rows'] == 3
    protocol = json.loads((output / 'protocol.json').read_text())
    assert len(protocol['schemas']) == 1
    assert all('schema' not in shard for shard in protocol['manifest']['64/train']['shards'])
    with sqlite3.connect(output / 'census.sqlite') as db:
        assert db.execute('SELECT occurrences FROM id_counts').fetchone()[0] == 3
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 3


def test_projection_preserves_ordinary_hash_rank(tmp_path):
    from passage.qualification import _ORDINARY, _describe, _stats
    root = tmp_path / 'data'
    write(root, 'train', [scene('a', 'train')])
    shard = root / '64x64' / 'train' / 'part-0.parquet'
    with pq.ParquetFile(shard) as parquet:
        full = next(parquet.iter_batches(columns=['id', 'metadata'])).to_pylist()[0]
        projected = next(parquet.iter_batches(columns=['id'] + [f'metadata.{name}' for name in _ORDINARY])).to_pylist()[0]
    locator = {'shard': '64x64/train/part-0.parquet', 'row_index': 0, 'row_group': 0, 'row_in_group': 0}
    a = _describe(full, 64, 'train', 17, locator, _stats())
    b = _describe(projected, 64, 'train', 17, locator, _stats())
    assert a['rank'] == b['rank']
    assert a['ordinary_metadata_sha256'] == b['ordinary_metadata_sha256']
