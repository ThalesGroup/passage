"""Publication metadata derives release claims from frozen output receipts."""
import json
import re

import pytest


def test_publication_metadata_rejects_unverified_or_blocked_release(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata
    from passage.export import ReleaseContext
    receipts = tmp_path / 'release-receipts'
    receipts.mkdir()
    (receipts / 'release-verification.json').write_text(json.dumps({
        'status': 'verified', 'data_status': 'verified', 'dataset_version': 'passage-successor-v1',
        'admission': {'status': 'blocked', 'unresolved_variants': 6}, 'resolutions': {}}))
    context = ReleaseContext(tmp_path / 'config.yaml', tmp_path / 'raw', tmp_path / 'calibrate.json',
        tmp_path, tmp_path, {64: 1}, 4, None, 64, ['astar_elevation', 'astar_slope', 'astar_energy'])
    with pytest.raises(ValueError, match='qualified'):
        write_successor_publication_metadata(context)


def _publication_fixture(tmp_path):
    import hashlib
    import zipfile
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    import yaml
    from rasterio.transform import from_origin
    from passage.calibrate import run_calibrate
    from passage.export import _build_release_context, _features
    from passage.successor_generation import semantic_hash

    root = tmp_path / 'release'
    receipts = root / 'release-receipts'
    receipts.mkdir(parents=True)
    tiles = tmp_path / 'tiles'
    tiles.mkdir()
    tile = tiles / 'N000E000_DSM.tif'
    with rasterio.open(tile, 'w', driver='GTiff', height=1, width=2, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(0, 1, 1, 1)) as raster:
        raster.write(np.array([[-12.25, 345.75]], dtype=np.float32), 1)
    calibration = tmp_path / 'calibrate'
    run_calibrate(calibration, tiles, num_cpus=0)
    calibration_file = calibration / 'calibrate.json'
    (receipts / 'calibration.json').write_bytes(calibration_file.read_bytes())
    solvers = [{'name': f'astar_{model}', 'cost_model': model, 'weight': 62.4}
               for model in ('elevation', 'slope_uphill', 'energy')]
    config = {'resolutions': [64], 'num_samples': {64: 4}, 'generation': {'dataset_version': 'passage-successor-v1'},
        'solvers': solvers, 'splits': {'ratios': {name: .25 for name in ('train', 'calibration', 'validation', 'test')}, 'seed': 42},
        'export': {'chunk_size': {64: 2}, 'sample': {'num_samples': 1}}}
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config))
    (receipts / 'resolved-config.yaml').write_bytes(config_file.read_bytes())
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    producer = b'fixture producer source\n'
    with zipfile.ZipFile(receipts / 'producer-source.zip', 'w') as archive:
        archive.writestr('src/passage/fixture.py', producer)
    provenance = {'dataset_version': 'passage-successor-v1', 'configuration': config,
        'config_sha256': sha(config_file), 'calibration_sha256': sha(calibration_file),
        'sources': {'0,0': {'name': tile.name, 'sha256': sha(tile)}},
        'source_snapshot': {'passage/fixture.py': hashlib.sha256(producer).hexdigest()},
        'nodata_policy': 'reject-source-crop', 'geographic_policy': {'version': 'source-tile-blocks-v1', 'block_degrees': 5, 'buffer_tiles': 1},
        'terrain_normalization': 'nearest-angular-center-v1', 'sample_seeding': 'blake2b(seed,resolution,index,split)',
        'tie_policy': 'float64-priority-then-row-major-node-strict-improvement-v1',
        'scene_attempts': 100, 'endpoint_policy': 'one-distinct-pair-per-accepted-crop-no-feasibility-selection'}
    provenance['provenance_id'] = semantic_hash(provenance)
    (receipts / 'generation-provenance.json').write_text(json.dumps(provenance))
    for suffix in ('', '_samples'):
        for split in config['splits']['ratios']:
            directory = root / f'64x64{suffix}' / split
            directory.mkdir(parents=True)
            features = _features(bool(suffix), [s['name'] for s in solvers])
            row = {'id': '0000000000', 'tensor': b'toy schema fixture', 'metadata_json': '{}', 'scene_receipt': '{}',
                   'metadata': {'index': 0, 'resolution': 64, 'split': split}}
            for solver in solvers:
                for variant in ('free', 'obstacles'):
                    row[f"path_{solver['name']}_{variant}"] = []
            pq.write_table(pa.Table.from_pylist([row], schema=features.arrow_schema), directory / 'data.parquet', compression='zstd')
    last = receipts / 'batch-64-0000000000-0000000003.json'
    last.write_text('{}')
    (receipts / 'state-64.json').write_text(json.dumps({'generation_provenance_id': provenance['provenance_id'],
        'config_sha256': sha(config_file), 'calibration_sha256': sha(calibration_file),
        'last_manifest': last.name, 'last_manifest_sha256': sha(last)}))
    for prefix in ('cohort', 'supervision'):
        (receipts / f'{prefix}-64.jsonl').write_text('fixture frozen roster\n')
    counts = {split: 1 for split in config['splits']['ratios']}
    summary = {'requested_scenes': 4, 'core_counts': counts, 'sample_counts': counts,
        'successful_route_variants': 23, 'proved_infeasible_variants': 1, 'unresolved_variants': 0,
        'scene_status_counts': {f'{split}:complete': 1 for split in counts},
        'variant_status_counts': {'train:ok': 5, 'train:infeasible': 1, **{f'{split}:ok': 6 for split in counts if split != 'train'}},
        'reference_status_counts': {'train:exact': 5, 'train:no_path': 1, **{f'{split}:exact': 6 for split in counts if split != 'train'}},
        'cohort_sha256': sha(receipts / 'cohort-64.jsonl'), 'supervision_sha256': sha(receipts / 'supervision-64.jsonl')}
    (receipts / 'release-verification.json').write_text(json.dumps({'status': 'verified', 'data_status': 'verified',
        'dataset_version': 'passage-successor-v1', 'scope': 'full', 'resolutions': {'64': summary},
        'admission': {'status': 'qualified', 'unresolved_variants': 0}}))
    raw = tmp_path / 'raw'
    raw.mkdir()
    return _build_release_context(raw, calibration_filepath=calibration_file, config_filepath=config_file, export_dirpath=root)


def test_successor_metadata_uses_actual_range_counts_and_separate_no_path_outcomes(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata, verify_successor_publication_metadata
    context = _publication_fixture(tmp_path)
    receipt = write_successor_publication_metadata(context)
    root = context.export_dirpath
    card = (root / 'README.md').read_text()
    assert '-12.25 m to 345.75 m' in card
    assert '23 successful exact routes' in card
    assert '1 independently proved no-path variants' in card
    assert 'not_run' in card
    assert 'sea level' not in card and '237 variants' not in card
    # The dataset card cites no commit revision (40-hex identifiers).
    assert re.search(r'\b[0-9a-fA-F]{40}\b', card) is None
    assert json.loads((root / 'croissant.json').read_text())['version'] == 'passage-successor-v1'
    assert receipt['status'] == 'verified'
    assert verify_successor_publication_metadata(root)['receipt_sha256'] == receipt['receipt_sha256']
    (root / 'README.md').write_text(card.replace('345.75', '8767.0'))
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_successor_publication_metadata(root)
    # Even a refreshed file hash cannot bless a scientifically stale range.
    import hashlib
    receipt_path = root / 'release-receipts/publication-metadata.json'
    stored = json.loads(receipt_path.read_text())
    stored['files_sha256']['README.md'] = hashlib.sha256((root / 'README.md').read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match='scientific metadata'):
        verify_successor_publication_metadata(root)


def test_successor_metadata_rejects_footer_count_different_from_verified_population(tmp_path):
    from passage.successor_publication import write_successor_publication_metadata
    context = _publication_fixture(tmp_path)
    path = context.export_dirpath / 'release-receipts/release-verification.json'
    receipt = json.loads(path.read_text())
    receipt['resolutions']['64']['sample_counts']['test'] = 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='footer count'):
        write_successor_publication_metadata(context)
