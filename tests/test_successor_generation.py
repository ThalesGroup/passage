"""Release generation must fail closed and bind labels to decoded terrain."""
import importlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def successor():
    assert importlib.util.find_spec('passage.successor_generation') is not None, 'successor generation contract missing'
    return importlib.import_module('passage.successor_generation')


def write_tile(path, data, nodata=None):
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1,
                       dtype=data.dtype, nodata=nodata, transform=from_origin(0, 1, 1, 1)) as dst:
        dst.write(data, 1)


def test_missing_and_nodata_terrain_rejected(tmp_path):
    s = successor()
    with pytest.raises(FileNotFoundError):
        s.read_source_window(tmp_path / 'missing.tif', 0, 0, 2, 2, 2)
    p = tmp_path / 'tile.tif'
    write_tile(p, np.array([[1, -9999], [2, 3]], dtype=np.float32), -9999)
    with pytest.raises(ValueError, match='nodata'):
        s.read_source_window(p, 0, 0, 2, 2, 2)
    with pytest.raises(ValueError, match='window'):
        s.read_source_window(p, 0, 1, 2, 2, 2)


def test_geographic_wrap_and_neighbor_policy():
    s = successor()
    policy = s.GeographicPolicy({'train': .90, 'validation': .04, 'calibration': .02, 'test': .04}, 42, 5, 0)
    assert policy.split_for_tile(0, 180) == policy.split_for_tile(0, -180)
    for lon in range(-180, 180):
        expected = len({policy.split_for_tile(0, lon), policy.split_for_tile(0, lon + 1)}) == 1
        assert policy.permits([(0, lon), (0, lon + 1)], policy.split_for_tile(0, lon)) == expected
    # Guard cells across a partition boundary reject the crop before labels exist.
    guarded = s.GeographicPolicy(policy.ratios, 42, 5, 1)
    boundary = next((lat, lon) for lat in range(-80, 80) for lon in range(-179, 179)
                    if policy.split_for_tile(lat, lon) != policy.split_for_tile(lat, lon+1))
    assert not guarded.permits([boundary], policy.split_for_tile(*boundary))


def test_receipt_rejects_missing_paths_and_wrong_provenance(tmp_path):
    s = successor()
    receipt = {'schema_version': s.SCENE_RECEIPT_VERSION, 'dataset_version': s.DATASET_VERSION,
               'index': 0, 'resolution': 2, 'split': 'train', 'scene_status': 'complete',
               'provenance_id': 'a', 'artifacts': {}, 'variants': {}}
    (tmp_path/'receipts').mkdir()
    (tmp_path/'receipts'/'0000000000.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provenance'):
        s.validate_scene_receipt(tmp_path, 0, expected_provenance_id='b')
    with pytest.raises(ValueError, match='six|artifact'):
        s.validate_scene_receipt(tmp_path, 0)


def scene_context(tmp_path):
    s = successor()
    from passage import generate
    config = {'generation': {'dataset_version': s.DATASET_VERSION},
              'obstacles': {'enabled': False},
              'solvers': [{'name': 'astar_'+name, 'cost_model': model, 'weight': weight,
                           'diagonal_movement': 'always'} for name,model,weight in
                          [('elevation','elevation',62.4), ('energy','energy',1125.3), ('slope','slope_uphill',278.0)]]}
    sources = {}
    for lat in range(-1,2):
        for lon in range(-1,2):
            p = tmp_path / f'{generate.lat_to_str(lat)}{generate.lon_to_str(lon)}_DSM.tif'
            data = (np.indices((128,128)).sum(axis=0) + 10).astype(np.float32)
            write_tile(p, data)
            sources[f'{lat},{lon}'] = {'path': str(p), 'name': p.name, 'sha256': s.file_hash(p)}
    return {'config': config, 'resolution': 64, 'output': str(tmp_path/'out'), 'sources': sources,
            'calibration': {'global_min': -1., 'global_max': 1000.}, 'seed': 42,
            'policy': s.GeographicPolicy({'train':1},42,5,0), 'candidates': {'train': [(0,0)]},
            'tile_size': 128, 'max_attempts': 10, 'max_expansions': None, 'provenance_id': 'fixture'}


def test_persisted_roundtrip_six_labels_and_missing_path_resume(tmp_path):
    s = successor()
    context = scene_context(tmp_path)
    result = s.generate_successor_scene(context, 0, 'train')
    assert result[2] == 'complete'
    root = tmp_path/'out'/'64'/'train'
    receipt = s.validate_scene_receipt(root, 0, 'fixture')
    assert len(receipt['variants']) == 6
    assert all(v['status'] == 'ok' and v['cost'] > 0 for v in receipt['variants'].values())
    original = s.semantic_hash(receipt)
    assert s.generate_successor_scene(context,0,'train')[2] == 'complete'
    assert s.semantic_hash(s.validate_scene_receipt(root,0,'fixture')) == original
    (root/receipt['artifacts']['path:astar_energy:free']['path']).unlink()
    with pytest.raises(ValueError, match='artifact'):
        s.generate_successor_scene(context,0,'train')


def _isolating_goal(shape, marker_positions):
    mask = np.zeros(shape,dtype=np.uint8)
    # Isolate goal: destination-only diagonals still blocked by all 8 neighbours.
    goal = marker_positions[1]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if (dy or dx) and 0 <= goal[0]+dy < shape[0] and 0 <= goal[1]+dx < shape[1]:
                mask[goal[0]+dy,goal[1]+dx] = 1
    mask[marker_positions[0]] = 0
    return mask


def test_disconnected_endpoints_are_rejected_and_redrawn_before_labels(tmp_path, monkeypatch):
    s = successor()
    from passage import generate
    context = scene_context(tmp_path)
    context['config']['obstacles'] = {'enabled': True, 'ratio_max': .1}
    calls = []
    def first_isolates(shape, *args, marker_positions=None, **kwargs):
        calls.append(marker_positions)
        if len(calls) == 1:
            return _isolating_goal(shape, marker_positions), .1
        return np.zeros(shape,dtype=np.uint8), .1
    monkeypatch.setattr(generate,'_generate_obstacles_mask',first_isolates)
    assert s.generate_successor_scene(context,0,'train')[2] == 'complete'
    receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',0)
    assert [a['reason'] for a in receipt['attempts']] == ['endpoints_disconnected_by_obstacles', 'accepted_input']
    assert all(v['status']=='ok' and v['cost'] > 0 for v in receipt['variants'].values())
    monkeypatch.setattr(generate,'_generate_obstacles_mask',
                        lambda shape, *a, marker_positions=None, **k: (_isolating_goal(shape, marker_positions), .1))
    assert s.generate_successor_scene(context,1,'train')[2] == 'rejected'
    receipt = json.loads((tmp_path/'out'/'64'/'train'/'receipts'/'0000000001.json').read_text())
    assert receipt['scene_status'] == 'rejected' and not receipt['artifacts']
    assert {a['reason'] for a in receipt['attempts']} == {'endpoints_disconnected_by_obstacles'}
    assert len(receipt['attempts']) == context['max_attempts']


def test_solver_contradicting_connectivity_fails_visibly(tmp_path, monkeypatch):
    s = successor()
    from passage import reference
    context = scene_context(tmp_path)
    proved = reference.ReferenceResult('no_path', (), None, 0, {}, 0.)
    monkeypatch.setattr(reference, 'solve_reference', lambda *a, **k: proved)
    with pytest.raises(RuntimeError, match='no path'):
        s.generate_successor_scene(context,0,'train')
    receipt = json.loads((tmp_path/'out'/'64'/'train'/'receipts'/'0000000000.json').read_text())
    assert receipt['scene_status'] == 'failed' and not receipt['artifacts']


def test_budget_outcomes_are_explicit_without_input_resampling(tmp_path):
    s = successor()
    context = scene_context(tmp_path)
    context['max_expansions'] = 0
    assert s.generate_successor_scene(context,1,'train')[2] == 'complete'
    receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',1)
    assert all(v['status']=='failed' and v['cost'] is None for v in receipt['variants'].values())


def test_native_coarsened_longitude_explicit_nearest_normalization(tmp_path):
    s = successor()
    p = tmp_path/'coarsened.tif'
    write_tile(p,np.arange(8,dtype=np.float32).reshape(4,2))
    expected = np.repeat(np.arange(8,dtype=np.float32).reshape(4,2),2,axis=1)
    np.testing.assert_array_equal(s.read_source_window(p,0,0,4,4,4), expected)
    np.testing.assert_array_equal(s.read_source_window(p,1,1,2,2,4), expected[1:3,1:3])


def test_serial_parallel_bounded_and_interrupted_replay(tmp_path, monkeypatch):
    s = successor()
    from passage import generate
    from passage.calibrate import run_calibrate
    import yaml
    context = scene_context(tmp_path)
    monkeypatch.setattr(generate, 'TILE_SIZE_PX',128)
    config = dict(context['config'], resolutions=[64], num_samples=4, splits={'ratios':{'train':1},'seed':42})
    config['generation'] = {'dataset_version':s.DATASET_VERSION,'seed':42,'max_scene_attempts':10,
                            'preview_images':{'enabled':False},'geographic_policy':{'block_degrees':5,'buffer_tiles':0}}
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    run_calibrate(str(tmp_path/'calibration'), str(tmp_path),num_cpus=0)
    calib = tmp_path/'calibration'/'calibrate.json'
    common = {'config_filepath':str(cfg),'tiles_dirpath':str(tmp_path),'calibration_filepath':str(calib),
              'resolution':64,'disable_images':True}
    serial = tmp_path/'serial'
    parallel = tmp_path/'parallel'
    resumed = tmp_path/'resumed'
    a = s.run_successor_generation(str(serial),num_cpus=0,**common)
    b = s.run_successor_generation(str(parallel),num_cpus=2,**common)
    first = s.run_successor_generation(str(resumed),num_cpus=0,scene_indices=range(0,2),**common)
    rest = s.run_successor_generation(str(resumed),num_cpus=2,scene_indices=range(2,4),**common)
    assert a['terminal_counts'] == b['terminal_counts'] == {'complete':4}
    assert first['indices'] == [0,1] and rest['indices'] == [2,3]
    for index in range(4):
        receipts = [s.validate_scene_receipt(root/'64'/'train',index,a['provenance_id'])
                    for root in (serial,parallel,resumed)]
        assert len({s.semantic_hash(r) for r in receipts}) == 1
    # Resuming no longer accepts changed config or source inventory identity.
    config['generation']['seed'] = 43
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='provenance'):
        s.run_successor_generation(str(serial),num_cpus=0,**common)


def test_uncommitted_installation_replays_identically(tmp_path):
    s = successor()
    context = scene_context(tmp_path)
    s.generate_successor_scene(context,0,'train')
    root = tmp_path/'out'/'64'/'train'
    original = s.validate_scene_receipt(root,0,'fixture')
    (root/'receipts'/'0000000000.json').unlink()
    # Simulate interruption after some files were installed but before commit.
    (root/original['artifacts']['path:astar_slope:free']['path']).write_text('0,0\n')
    assert s.generate_successor_scene(context,0,'train')[2] == 'complete'
    assert s.semantic_hash(s.validate_scene_receipt(root,0,'fixture')) == s.semantic_hash(original)


def test_failed_write_is_retained_and_never_reported_complete(tmp_path,monkeypatch):
    s = successor()
    context = scene_context(tmp_path)
    def fail(*args,**kwargs):
        raise OSError('test write failure')
    monkeypatch.setattr(s,'save_tensor',fail)
    with pytest.raises(OSError,match='write failure'):
        s.generate_successor_scene(context,0,'train')
    receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',0,'fixture')
    assert receipt['scene_status']=='failed' and receipt['artifacts']=={}
    assert all(v['status']=='failed' and v['cost'] is None for v in receipt['variants'].values())


def test_sidecar_added_after_authentication_cannot_change_source_validity(tmp_path):
    s = successor()
    p = tmp_path/'N000E000_DSM.tif'
    write_tile(p,np.ones((4,4),dtype=np.float32))
    s.enforce_no_external_raster_dependencies(p)
    original = s.file_hash(p)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(p,'r+') as src:
            src.write_mask(np.zeros((4,4),dtype=np.uint8))
    assert s.file_hash(p) == original  # DSM-only provenance misses this external mask.
    with pytest.raises(ValueError,match='external_raster_dependency'):
        s.read_source_window(p,0,0,4,4,4)


def test_frozen_calibration_change_is_observable_before_scene_commit(tmp_path,monkeypatch):
    s = successor()
    context = scene_context(tmp_path)
    control = tmp_path/'calibration.json'
    control.write_text('{"frozen":true}')
    context['control_files'] = {str(control):{'sha256':s.file_hash(control),'fingerprint':s._fingerprint(control)}}
    original_save = s.save_tensor
    def change_control(*args,**kwargs):
        original_save(*args,**kwargs)
        control.write_text('{"frozen":false}')
    monkeypatch.setattr(s,'save_tensor',change_control)
    with pytest.raises(RuntimeError,match='calibration changed'):
        s.generate_successor_scene(context,0,'train')
    receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',0,'fixture')
    assert receipt['scene_status']=='failed' and not receipt['artifacts']


def test_live_header_cache_still_rejects_new_external_mask(tmp_path,monkeypatch):
    s = successor()
    from passage import generate
    monkeypatch.setattr(generate,'TILE_SIZE_PX',4)
    p = tmp_path/'N000E000_DSM.tif'
    write_tile(p,np.ones((4,4),dtype=np.float32))
    cfg = tmp_path/'cfg.json'
    cfg.write_text('{}')
    cal = tmp_path/'cal.json'
    cal.write_text('{}')
    policy = s.GeographicPolicy({'train':1},42,5,0)
    s.prepare_provenance(cfg,cal,tmp_path,{},policy)
    def must_not_reopen(*args,**kwargs):
        raise AssertionError('unchanged authenticated DSM header reopened')
    monkeypatch.setattr(s,'enforce_no_external_raster_dependencies',must_not_reopen)
    s.prepare_provenance(cfg,cal,tmp_path,{},policy)
    (tmp_path/(p.name+'.msk')).write_bytes(b'external')
    with pytest.raises(ValueError,match='external_raster_dependency'):
        s.prepare_provenance(cfg,cal,tmp_path,{},policy)


def test_holdout_bytes_bypass_legacy_cache_and_changed_mask_blocks_resume(tmp_path,monkeypatch):
    s = successor()
    from passage import generate
    from passage.calibrate import run_calibrate
    from PIL import Image
    import yaml
    context = scene_context(tmp_path)
    monkeypatch.setattr(generate,'TILE_SIZE_PX',128)
    mask = tmp_path/'holdout.png'
    Image.fromarray(np.zeros((181,361),dtype=np.uint8)).save(mask)
    generate.load_hold_out_mask(mask)  # Populate legacy path-only cache deliberately.
    config = dict(context['config'], resolutions=[64],num_samples=1,
                  splits={'ratios':{'train':1},'seed':42},hold_out={'enabled':True,'mask_path':str(mask)})
    config['generation'] = {'dataset_version':s.DATASET_VERSION,'seed':42,'max_scene_attempts':10,
                            'geographic_policy':{'block_degrees':5,'buffer_tiles':0}}
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    run_calibrate(str(tmp_path/'calibration'),str(tmp_path),num_cpus=0)
    common = {'config_filepath':str(cfg),'tiles_dirpath':str(tmp_path),
              'calibration_filepath':str(tmp_path/'calibration'/'calibrate.json'),
              'resolution':64,'num_cpus':0,'disable_images':True}
    output = tmp_path/'output'
    s.run_successor_generation(str(output),**common)
    manifest = json.loads((output/'generation_manifest.json').read_text())
    assert manifest['holdout_mask_file_sha256'] == s.file_hash(mask)
    archived = output/'generation-input-snapshot'/'holdout-mask'/mask.name
    assert s.file_hash(archived) == s.file_hash(mask)
    Image.fromarray(np.full((181,361),255,dtype=np.uint8)).save(mask)
    with pytest.raises(ValueError,match='provenance'):
        s.run_successor_generation(str(output),**common)
    mask.unlink()
    with pytest.raises(FileNotFoundError):
        s.run_successor_generation(str(output),**common)


@pytest.mark.parametrize('field',['source_windows','sample_seed','attempts'])
def test_complete_receipt_requires_replay_fields(tmp_path,field):
    s = successor()
    context = scene_context(tmp_path)
    s.generate_successor_scene(context,0,'train')
    root = tmp_path/'out'/'64'/'train'
    receipt = s.validate_scene_receipt(root,0,'fixture')
    receipt.pop(field)
    s._write_json(root/'receipts'/'0000000000.json',receipt)
    with pytest.raises(ValueError,match='replay|source|seed|attempt'):
        s.validate_scene_receipt(root,0,'fixture')


def test_parallel_worker_native_exit_fails_visibly(tmp_path):
    import subprocess
    import sys
    context = scene_context(tmp_path)
    minimal = {key:context[key] for key in ('output','resolution','config','provenance_id','calibration','seed')}
    driver = tmp_path/'worker_death.py'
    driver.write_text('''import os
import json
from passage import successor_generation as s

def native_exit(task):
    os._exit(19)

if __name__ == '__main__':
    s._task = native_exit
    try:
        context = json.loads(CONTEXT_JSON)
        list(s._execute_parallel(context, iter([(0,'train',False),(1,'train',False)]), 2))
    except Exception as exc:
        print(type(exc).__name__,flush=True)
        raise SystemExit(0 if type(exc).__name__ == 'BrokenProcessPool' else 3)
    raise SystemExit(4)
'''.replace('CONTEXT_JSON',repr(json.dumps(minimal))))
    result = subprocess.run([sys.executable,str(driver)],capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'BrokenProcessPool' in result.stdout

    s = successor()
    for index in (0,1):
        receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',index,'fixture')
        assert receipt['scene_status'] == 'failed' and receipt['failure']['type']=='BrokenProcessPool'
        assert all(v['status']=='failed' and v['cost'] is None for v in receipt['variants'].values())

    # Ordinary same-identity resume retries infrastructure failure, preserving the failed attempt.
    for index in (0,1):
        assert s.generate_successor_scene(context,index,'train')[2] == 'complete'
        receipt = s.validate_scene_receipt(tmp_path/'out'/'64'/'train',index,'fixture')
        assert receipt['infrastructure_attempt'] == 2 and len(receipt['attempt_history']) == 1
        archived = tmp_path/'out'/'64'/'train'/receipt['attempt_history'][0]['receipt']
        assert json.loads(archived.read_text())['failure']['type'] == 'BrokenProcessPool'


def _session_inputs(tmp_path, monkeypatch, *, resolutions=(64,), count=4):
    from passage import generate
    from passage.calibrate import run_calibrate
    import yaml
    context = scene_context(tmp_path)
    monkeypatch.setattr(generate, 'TILE_SIZE_PX', 128)
    config = dict(context['config'], resolutions=list(resolutions), num_samples=count,
                  splits={'ratios': {'train': 1}, 'seed': 42})
    config['generation'] = {'dataset_version': 'passage-successor-v1', 'seed': 42,
                            'max_scene_attempts': 10,
                            'geographic_policy': {'block_degrees': 5, 'buffer_tiles': 0}}
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    run_calibrate(str(tmp_path/'calibration'), str(tmp_path), num_cpus=0)
    return dict(config_filepath=cfg, tiles_dirpath=tmp_path,
                calibration_filepath=tmp_path/'calibration'/'calibrate.json')


def _worker_environment():
    import os
    from rasterio._env import get_gdal_config
    return os.getpid(), get_gdal_config('GDAL_CACHEMAX'), os.environ.get('NUMBA_NUM_THREADS')


def _blocking_export_job(started):
    from pathlib import Path
    import threading
    Path(started).touch()
    threading.Event().wait()


def _session_jobs(tmp_path, monkeypatch):
    # pytest's importlib mode does not make tests.* importable in spawned Python.
    # Give real external jobs an ordinary temporary, importable module.
    import importlib
    import inspect
    name = 'session_jobs_'+tmp_path.name.replace('-', '_')
    (tmp_path/(name+'.py')).write_text(inspect.getsource(_worker_environment)+'\n'+
                                     inspect.getsource(_blocking_export_job))
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(name)


def test_session_workers_survive_generation_and_export_boundaries(tmp_path, monkeypatch):
    """Recreating the pool or the authenticated roster between chunks is a regression."""
    s = successor()
    assert hasattr(s, 'SuccessorGenerationSession'), 'persistent generation session missing'
    common = _session_inputs(tmp_path, monkeypatch)
    jobs = _session_jobs(tmp_path, monkeypatch)
    serial, persistent = tmp_path/'serial', tmp_path/'persistent'
    expected = s.run_successor_generation(serial, num_cpus=0, resolution=64, **common)
    with s.SuccessorGenerationSession(persistent, num_cpus=2, **common) as session:
        context = session.prepare_resolution(64)
        progress = []
        first = session.generate(range(2), progress_callback=lambda completed, total: progress.append((completed, total)))
        assert progress == [(1, 2), (2, 2)]
        executor = session.executor
        processes = list(executor._processes.values())
        pid, cache_bytes, threads = executor.submit(jobs._worker_environment).result(timeout=10)
        assert pid in {process.pid for process in processes}
        assert cache_bytes == 256*1024**2 and threads == '1'
        assert session.prepare_resolution(64) is context
        assert session.ensure_executor(1) is executor
        rest = session.generate(range(2, 4))
        later_pid, _, _ = executor.submit(jobs._worker_environment).result(timeout=10)
        assert session.executor is executor
        assert later_pid in {process.pid for process in processes}
        assert all(process.is_alive() for process in processes)
        assert first['terminal_counts'] == rest['terminal_counts'] == {'complete': 2}
        assert 'indices' not in first and first['selected_samples'] == 2
        for index in range(4):
            actual = s.validate_scene_receipt(persistent/'64'/'train', index, expected['provenance_id'])
            reference = s.validate_scene_receipt(serial/'64'/'train', index, expected['provenance_id'])
            assert actual == reference
        progress.clear()
        resumed = session.generate(range(2), progress_callback=lambda completed, total: progress.append((completed, total)))
        assert resumed['timing']['generated_this_run'] == 0
        assert progress == [(1, 2), (2, 2)]
    assert all(not process.is_alive() and process.exitcode is not None for process in processes)


def test_session_exception_stops_and_joins_running_export_workers(tmp_path, monkeypatch):
    """Cancellation must release owned native work before the scratch owner exits."""
    import time
    s = successor()
    assert hasattr(s, 'SuccessorGenerationSession'), 'persistent generation session missing'
    common = _session_inputs(tmp_path, monkeypatch)
    jobs = _session_jobs(tmp_path, monkeypatch)
    started = tmp_path/'worker-started'
    with pytest.raises(KeyboardInterrupt):
        with s.SuccessorGenerationSession(tmp_path/'out-session', num_cpus=2, **common) as session:
            session.prepare_resolution(64)
            session.generate(range(2))
            session.executor.submit(jobs._blocking_export_job, str(started))
            processes = list(session.executor._processes.values())
            deadline = time.monotonic()+10
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert started.exists(), 'blocking worker did not start'
            before = time.monotonic()
            raise KeyboardInterrupt('controlled cancellation')
    assert time.monotonic()-before < 10
    assert all(not process.is_alive() and process.exitcode is not None for process in processes)
    assert s.validate_scene_receipt(tmp_path/'out-session'/'64'/'train', 0)['scene_status'] == 'complete'


def test_session_native_export_exit_joins_shared_pool_and_keeps_scene_receipts(tmp_path, monkeypatch):
    import os
    from concurrent.futures.process import BrokenProcessPool
    s = successor()
    common = _session_inputs(tmp_path, monkeypatch)
    output = tmp_path/'native-export-out'
    with pytest.raises(BrokenProcessPool):
        with s.SuccessorGenerationSession(output, num_cpus=2, **common) as session:
            session.prepare_resolution(64)
            session.generate(range(2))
            committed = [s.validate_scene_receipt(output/'64'/'train', index) for index in (0, 1)]
            processes = list(session.executor._processes.values())
            session.executor.submit(os._exit, 19).result(timeout=10)
    assert all(not process.is_alive() and process.exitcode is not None for process in processes)
    assert [s.validate_scene_receipt(output/'64'/'train', index) for index in (0, 1)] == committed
    with s.SuccessorGenerationSession(output, num_cpus=0, **common) as resumed:
        resumed.prepare_resolution(64)
        assert resumed.generate(range(4))['terminal_counts'] == {'complete': 4}


def test_session_progress_failure_aborts_pool_without_losing_commits(tmp_path, monkeypatch):
    s = successor()
    common = _session_inputs(tmp_path, monkeypatch)
    session = s.SuccessorGenerationSession(tmp_path/'callback-out', num_cpus=2, **common)
    session.prepare_resolution(64)
    processes = []

    def fail_progress(completed, total):
        assert completed == 1 and total == 4
        processes.extend(session.executor._processes.values())
        raise RuntimeError('progress recorder failed')

    with pytest.raises(RuntimeError, match='progress recorder failed'):
        session.generate(range(4), progress_callback=fail_progress)
    assert processes and all(not process.is_alive() for process in processes)
    assert session.executor is None
    receipts = list((tmp_path/'callback-out'/'64'/'train'/'receipts').glob('*.json'))
    assert receipts and any(s.validate_scene_receipt(path.parent.parent, int(path.stem))['scene_status'] == 'complete'
                            for path in receipts)
    # A same-identity fresh session replays cancelled inputs with their original seed.
    with s.SuccessorGenerationSession(tmp_path/'callback-out', num_cpus=0, **common) as resumed:
        resumed.prepare_resolution(64)
        assert resumed.generate(range(4))['terminal_counts'] == {'complete': 4}


def test_session_resolution_change_joins_old_pool_and_freezes_new_context(tmp_path, monkeypatch):
    s = successor()
    common = _session_inputs(tmp_path, monkeypatch, resolutions=(32, 64))
    with s.SuccessorGenerationSession(tmp_path/'multi-resolution', num_cpus=2, **common) as session:
        original = session.prepare_resolution(64)
        session.generate(range(1))
        processes = list(session.executor._processes.values())
        changed = session.prepare_resolution(32)
        assert all(not process.is_alive() for process in processes)
        assert changed['resolution'] == 32 and original['resolution'] == 64
        assert session.generate(range(1))['terminal_counts'] == {'complete': 1}
        with pytest.raises(ValueError, match='roster'):
            session.prepare_resolution(32, requested_samples=5)


def test_scene_metadata_safe_parser_matches_legacy_and_rejects_python_tags(tmp_path):
    import yaml
    s = successor()
    context = scene_context(tmp_path)
    s.generate_successor_scene(context, 0, 'train')
    root = tmp_path/'out'/'64'/'train'
    receipt = s.validate_scene_receipt(root, 0)
    metadata = root/receipt['artifacts']['metadata']['path']
    assert yaml.safe_load(metadata.read_text()) == yaml.load(
        metadata.read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    metadata.write_text(metadata.read_text()+'unsafe: !!python/object/apply:builtins.print [forbidden]\n')
    receipt['artifacts']['metadata']['sha256'] = s.file_hash(metadata)
    s._write_json(root/'receipts'/'0000000000.json', receipt)
    with pytest.raises(yaml.constructor.ConstructorError):
        s.validate_scene_receipt(root, 0)


def test_session_freezes_code_identity_before_first_resolution(tmp_path, monkeypatch):
    """A later resolution must not bless a changed producer's new fingerprint."""
    from pathlib import Path
    s = successor()
    common = _session_inputs(tmp_path, monkeypatch)
    with s.SuccessorGenerationSession(tmp_path/'frozen-code', num_cpus=0, **common) as session:
        code_path = session.source_root/'passage'/'reference.py'
        fingerprint = s._fingerprint

        def changed_identity(path):
            result = fingerprint(path)
            if Path(path) == code_path:
                result[-1] += 1
            return result

        # Simulate a stat identity change without changing the shared producers
        # that concurrent work or a running test may have imported.
        monkeypatch.setattr(s, '_fingerprint', changed_identity)
        with pytest.raises(RuntimeError, match='generating source changed'):
            session.prepare_resolution(64)


@pytest.mark.parametrize('change', ['terrain', 'new-tile', 'sidecar', 'calibration'])
def test_session_rechecks_authenticated_inputs_between_storage_stages(tmp_path, monkeypatch, change):
    s = successor()
    common = _session_inputs(tmp_path, monkeypatch)
    with s.SuccessorGenerationSession(tmp_path/'changing-inputs', num_cpus=0, **common) as session:
        session.prepare_resolution(64)
        session.generate(range(1))
        tile = tmp_path/'N000E000_DSM.tif'
        if change == 'terrain':
            with tile.open('ab') as stream:
                stream.write(b'changed')
        elif change == 'new-tile':
            write_tile(tmp_path/'N002E002_DSM.tif', np.ones((128, 128), dtype=np.float32))
        elif change == 'sidecar':
            tile.with_name(tile.name+'.msk').write_bytes(b'new external input')
        else:
            calibration = common['calibration_filepath']
            calibration.chmod(0o600)  # The real producer seals its receipt read-only.
            calibration.write_bytes(calibration.read_bytes()+b' ')
        with pytest.raises((RuntimeError, ValueError), match='changed|external_raster_dependency'):
            session.generate(range(1, 2))
