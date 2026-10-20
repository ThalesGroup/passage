"""Stage paper-style HF presentation assets without changing sealed Parquet PNGs.

Read the first validation preview at each resolution, without selection by path
quality. Authenticate each source shard against the admitted export manifest.
No solver, model, dataset rewrite, registration or upload is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from trajectory_style import STYLE_VERSION, render_cost_comparison

ROOT = Path(__file__).resolve().parents[1]
# Local PASSAGE v1.0 dataset root (the sealed upload folder); set PASSAGE_DATASET_ROOT to its path.
DATA = Path(os.environ.get('PASSAGE_DATASET_ROOT', ROOT / 'dataset'))
SIDES = (64, 128, 256, 512, 1024, 2048, 4096)
KEYS = [f'path_astar_{cost}_{regime}' for regime in ('free', 'obstacles')
        for cost in ('elevation', 'slope', 'energy')]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(output):
    output = output.resolve()
    if output == DATA or output.is_relative_to(DATA):
        raise ValueError('Presentation outputs must remain outside the sealed dataset')
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    completion_path = ROOT / 'outputs/regeneration/metadata-recovery-v3-20261013/completion.json'
    pin_path = ROOT / 'outputs/regeneration/metadata-monitor-v6-20261013/completion-pin.json'
    if digest(completion_path) != json.loads(pin_path.read_text())['completion_sha256']:
        raise ValueError('Admitted completion pin changed')
    completion = json.loads(completion_path.read_text())
    manifest_path = DATA / 'notebook-export-manifest.json'
    expected = completion['recovery_result']['publication_metadata_admission']['input_sha256']['notebook-export-manifest.json']
    if digest(manifest_path) != expected:
        raise ValueError('Admitted source export manifest changed')
    manifest = json.loads(manifest_path.read_text())
    records = []
    for side in SIDES:
        source = sorted((DATA / f'{side}x{side}_samples/validation').glob('*.parquet'))[0]
        relative = str(source.relative_to(DATA))
        if digest(source) != manifest['inputs'][relative]['sha256']:
            raise ValueError('Source shard identity changed: ' + relative)
        row = next(pq.ParquetFile(source).iter_batches(batch_size=1, columns=['id', 'metadata', 'tensor', *KEYS])).to_pylist()[0]
        routes = {key: hashlib.sha256(np.asarray(row[key], dtype='<i4').tobytes()).hexdigest() for key in KEYS}
        figure = render_cost_comparison(row, title='PASSAGE — same terrain, different routing costs')
        files = {}
        for suffix in ('png', 'pdf'):
            path = assets / f'sample-r{side}.{suffix}'
            figure.savefig(path, dpi=200, bbox_inches='tight', facecolor='white',
                           metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None)
            files[path.name] = digest(path)
        plt.close(figure)
        if routes != {key: hashlib.sha256(np.asarray(row[key], dtype='<i4').tobytes()).hexdigest() for key in KEYS}:
            raise ValueError('Presentation modified supplied path vertices')
        record = {'resolution': side, 'sample_id': row['id'], 'split': 'validation', 'source_row': 0,
                  'source_shard': relative, 'source_sha256': manifest['inputs'][relative]['sha256'],
                  'style': STYLE_VERSION, 'route_sha256_int32_row_col': routes, 'outputs': files,
                  'route_optimality_revalidated': False}
        (assets / f'sample-r{side}.provenance.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(json.dumps({'resolution': side, 'sample_id': row['id'], 'status': 'rendered'}), flush=True)
    for name in ('banner.png', 'download.png', 'hold_out_mask.png'):
        shutil.copyfile(ROOT / 'assets' / name, assets / name)
    # Notebook exports can import the same profile without a sibling paper repo.
    shutil.copyfile(Path(__file__).with_name('trajectory_style.py'), output / 'trajectory_style.py')
    for suffix in ('png', 'pdf'):
        shutil.copyfile(assets / f'sample-r256.{suffix}', assets / f'sample.{suffix}')
    (assets / 'sample.provenance.json').write_text(json.dumps(records[2], indent=2) + '\n')
    result = {'status': 'presentation_assets_generated_not_uploaded', 'style': STYLE_VERSION,
              'style_source_sha256': digest(Path(__file__).with_name('trajectory_style.py')),
              'renderer_sha256': digest(__file__), 'completion_sha256': digest(completion_path),
              'manifest_sha256': expected, 'selection': 'first row of first validation preview shard at each resolution; no filtering',
              'records': records, 'sealed_dataset_modified': False, 'embedded_parquet_PNGs_replaced': False,
              'new_routes_or_predictions_computed': False, 'upload': False}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/trajectory-style-v1-20261017/hf-presentation')
    args = parser.parse_args()
    render(args.output)
