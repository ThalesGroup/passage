"""Receipt-derived publication files for the released dataset; source documents stay intact."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import yaml

from passage.notebook_export import write_staged_bytes

if TYPE_CHECKING:
    from passage.export import ReleaseContext

VERSION = 'passage-successor-v1'
RECEIPT = 'release-receipts/publication-metadata.json'
FACTS = 'release-receipts/publication-facts.json'
SPLITS = ('train', 'calibration', 'validation', 'test')


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _facts(root: Path) -> tuple[dict, dict[str, str]]:
    """Check scientific identities, counts and source controls before drafting claims."""
    verification = _json(root / 'release-receipts/release-verification.json')
    if (verification.get('dataset_version') != VERSION or verification.get('data_status') != 'verified'
            or verification.get('status') != 'verified' or verification.get('admission', {}).get('status') != 'qualified'
            or verification['admission'].get('unresolved_variants') != 0):
        raise ValueError('Release publication requires a verified, qualified release without unresolved variants')
    from passage.calibrate import validate_calibration_receipt
    from passage.regenerate import _split_counts
    from passage.successor_generation import semantic_hash
    from passage_eval.protocol.contracts import GraphSpec

    controls = ['release-receipts/release-verification.json', 'release-receipts/resolved-config.yaml',
                'release-receipts/calibration.json', 'release-receipts/generation-provenance.json',
                'release-receipts/producer-source.zip']
    if (root / 'notebook-export-manifest.json').is_file():
        controls.append('notebook-export-manifest.json')
    config = yaml.safe_load((root / controls[1]).read_text())
    calibration = _json(root / controls[2])
    low, high = validate_calibration_receipt(calibration)
    provenance = _json(root / controls[3])
    provenance_id = provenance.get('provenance_id')
    if (provenance.get('dataset_version') != VERSION or config.get('generation', {}).get('dataset_version') != VERSION
            or provenance.get('config_sha256') != _hash(root / controls[1])
            or provenance.get('calibration_sha256') != _hash(root / controls[2])
            or provenance.get('configuration') != json.loads(json.dumps(config))
            or semantic_hash({**{k: v for k, v in provenance.items() if k not in ('provenance_id', 'configuration')},
                              'configuration': config}) != provenance_id):
        raise ValueError('Release publication generation/configuration/calibration identity mismatch')
    if provenance.get('nodata_policy') != 'reject-source-crop':
        raise ValueError('Unsupported release nodata policy')
    sources = {item['name']: item['sha256'] for item in provenance['sources'].values()}
    calibrated_sources = {item['filename']: item['sha256'] for item in calibration['source_inventory']}
    if sources != calibrated_sources:
        raise ValueError('Release publication calibration/source inventory mismatch')
    with zipfile.ZipFile(root / controls[4]) as archive:
        for path, expected in provenance['source_snapshot'].items():
            if hashlib.sha256(archive.read(f'src/{path}')).hexdigest() != expected:
                raise ValueError('Release publication archived producer differs from generating source')
    from passage.export import _features
    graphs = {item['name']: GraphSpec(cost_model=item['cost_model'], cost_weight=item['weight']).to_dict()
              for item in config['solvers']}
    if len(graphs) != 3:
        raise ValueError('Release publication requires three objective graphs')
    resolutions, configurations = {}, []
    for key, summary in sorted(verification['resolutions'].items(), key=lambda item: int(item[0])):
        resolution = int(key)
        configured = config['num_samples']
        expected_total = int(configured.get(resolution, configured.get(key, 0))) if isinstance(configured, dict) else int(configured)
        if verification.get('scope') == 'full' and (summary['requested_scenes'] != expected_total
                or summary['core_counts'] != _split_counts(expected_total, config['splits']['ratios'])):
            raise ValueError('Release publication core counts differ from configured full population')
        if (summary['unresolved_variants'] != 0 or summary['successful_route_variants'] + summary['proved_infeasible_variants']
                != 6 * summary['requested_scenes']):
            raise ValueError('Release publication reference/terminal denominator mismatch')
        for suffix, count_key in (('', 'core_counts'), ('_samples', 'sample_counts')):
            name = f'{resolution}x{resolution}{suffix}'
            configurations.append(name)
            for split in SPLITS:
                shards = sorted((root / name / split).glob('*.parquet'))
                count = 0
                if not shards:
                    raise ValueError(f'Missing release publication split: {name}/{split}')
                for shard in shards:
                    with pq.ParquetFile(shard) as parquet:
                        count += parquet.metadata.num_rows
                        if not parquet.schema_arrow.equals(_features(bool(suffix), list(graphs)).arrow_schema, check_metadata=False):
                            raise ValueError('Release publication Arrow schema differs from declared card features')
                        images = [column for column in parquet.schema_arrow.names if column.startswith('image_')]
                        if bool(images) != bool(suffix):
                            raise ValueError('Release publication core/preview image schema mismatch')
                if count != summary[count_key].get(split):
                    raise ValueError('Release publication footer count differs from verification')
                if suffix and verification.get('scope') == 'full' and count != min(64, int(config['export']['sample']['num_samples'])):
                    raise ValueError('Release publication preview count differs from configured sample size')
        state_relative = f'release-receipts/state-{resolution}.json'
        state = _json(root / state_relative)
        if (state.get('generation_provenance_id') != provenance_id or state.get('config_sha256') != provenance['config_sha256']
                or state.get('calibration_sha256') != provenance['calibration_sha256']):
            raise ValueError('Release publication continuation provenance mismatch')
        last_relative = f"release-receipts/{state['last_manifest']}"
        if _hash(root / last_relative) != state['last_manifest_sha256']:
            raise ValueError('Release publication final batch manifest mismatch')
        controls.extend([state_relative, last_relative])
        for prefix in ('cohort', 'supervision'):
            relative = f'release-receipts/{prefix}-{resolution}.jsonl'
            if _hash(root / relative) != summary[f'{prefix}_sha256']:
                raise ValueError('Release publication qualified cohort identity mismatch')
            controls.append(relative)
        resolutions[key] = summary
    if not resolutions:
        raise ValueError('Release publication requires verified resolution populations')
    facts = {'dataset_version': VERSION, 'scope': verification['scope'], 'configurations': configurations,
        'resolutions': resolutions, 'core_scenes': sum(s['requested_scenes'] for s in resolutions.values()),
        'preview_rows': sum(sum(s['sample_counts'].values()) for s in resolutions.values()),
        'route_variants': sum(s['requested_scenes'] * 6 for s in resolutions.values()),
        'successful_route_variants': sum(s['successful_route_variants'] for s in resolutions.values()),
        'proved_infeasible_variants': sum(s['proved_infeasible_variants'] for s in resolutions.values()),
        'calibration': {'min_m': low, 'max_m': high, 'sha256': provenance['calibration_sha256'],
                        'source_inventory_sha256': calibration['source_inventory_sha256']},
        'generation_provenance_id': provenance_id, 'graphs': graphs,
        'geographic_policy': provenance['geographic_policy'], 'nodata_policy': provenance['nodata_policy'],
        'terrain_normalization': provenance['terrain_normalization'], 'sample_seeding': provenance['sample_seeding'],
        'tie_policy': provenance['tie_policy'], 'scene_attempts': provenance['scene_attempts'],
        'endpoint_policy': provenance['endpoint_policy'], 'split_ratios': config['splits']['ratios'],
        'split_seed': config['splits']['seed'], 'learned_results': 'not_run', 'upload': 'not_run'}
    return facts, {path: _hash(root / path) for path in controls}


def _description(facts: dict) -> str:
    # The published card, datasheet and Croissant metadata embed this exact text, and
    # verification recomputes it; keep it unchanged for the sealed release receipts.
    return (f"PASSAGE successor {VERSION} contains {facts['core_scenes']:,} core scenes and "
        f"{facts['route_variants']:,} objective/obstacle variants; {facts['preview_rows']:,} separate preview rows "
        "do not add independent training observations. Terrain and labels use persisted float32 inputs decoded "
        "before authoritative float64 graph solves. Missing/nodata source crops are rejected. "
        f"The qualified outcomes include {facts['successful_route_variants']:,} successful exact routes and "
        f"{facts['proved_infeasible_variants']:,} independently proved no-path variants. Learned results are not_run.")


def write_successor_publication_metadata(context: ReleaseContext, *, repo_id: str = 'thalesgroup/passage') -> dict:
    """Write the dataset card, datasheet, Croissant metadata and a scientific content receipt."""
    root = context.export_dirpath.resolve()
    facts, inputs = _facts(root)
    if (_hash(context.config_file) != inputs['release-receipts/resolved-config.yaml']
            or _hash(context.calibration_file) != facts['calibration']['sha256']):
        raise ValueError('Release publication current controls differ from archived inputs')
    from passage.export import _features, _generate_huggingface_yaml_header, _repo_root
    source_root = _repo_root()
    description = _description(facts)
    rows = ['| Resolution | Split | Core scenes | Preview scenes |', '| --- | --- | ---: | ---: |']
    for resolution, summary in facts['resolutions'].items():
        for split in SPLITS:
            rows.append(f"| {resolution} | {split} | {summary['core_counts'][split]:,} | {summary['sample_counts'][split]:,} |")
    evidence = ('[release verification](release-receipts/release-verification.json), '
        '[generation provenance](release-receipts/generation-provenance.json), '
        '[publication facts](release-receipts/publication-facts.json) and '
        '[metadata hashes](release-receipts/publication-metadata.json)')
    body = (f"# PASSAGE v1.0\n\nPublication upload: **not_run**.\n\n{description}\n\n"
        '## Population and inputs\n\n' + '\n'.join(rows) + '\n\n'
        f"The complete calibration range is **{facts['calibration']['min_m']!r} m to {facts['calibration']['max_m']!r} m**. "
        'See [calibrate.json](calibration/calibrate.json) for the authenticated source inventory and validity policy. '
        'The float32 tensor stores normalized elevation, start/goal markers and a binary obstacle channel. '
        'Core Parquets contain no image columns. Previews depict the lowest admitted complete scene IDs per split; '
        'they are inspection examples rather than population evidence. Full replay metadata is retained in `metadata_json` '
        'and `scene_receipt`; ordinary graph parameters and terminal path statuses are also structured metadata.\n\n'
        '## Graph and label contract\n\n'
        'The routing graph is eight-connected with destination-only diagonal traversal, blocked obstacles and a fixed '
        '30 m graph distance convention. Elevation/slope/energy costs are routing proxies on the JAXA ALOS AW3D30 '
        'digital surface model, which includes canopy/building surfaces; 30 m graph spacing does not certify actual '
        'ground spacing. Calibration and objective weights remain float64 in exported metadata.\n\n'
        'Successful `ok` paths are legal and exact under the frozen GraphSpec. `infeasible` means the reference solve '
        'proved no path; its ordered route is empty and cost is null. Failed/censored outcomes retain explicit terminal '
        'records and block qualification. Successful-path supervision uses separate supervision rosters; infeasible '
        'tasks do not become zero-cost successful examples. Tied optima may have different path geometry.\n\n'
        '```json\n' + json.dumps(facts['graphs'], indent=2, sort_keys=True) + '\n```\n\n'
        '## Geography and replay\n\n'
        'Source-cell windows, including neighboring tiles and wrapped longitude, are retained in scene receipts. '
        'The frozen source tile partition and guard policy is recorded below; complete crop support must belong '
        'to its assigned split. Missing tiles, nodata and incomplete source windows are rejected rather than padded.\n\n'
        '```json\n' + json.dumps(facts['geographic_policy'], indent=2, sort_keys=True) + '\n```\n\n'
        f"Sample seeding: `{facts['sample_seeding']}`. Normalization: `{facts['terrain_normalization']}`. "
        f"Tie policy: `{facts['tie_policy']}`. Maximum scene attempts: {facts['scene_attempts']}. "
        f"Endpoint policy: `{facts['endpoint_policy']}`. All requested scene/variant outcomes remain in release receipts.\n\n"
        'Examples are the verified `_samples` Parquets and executed display notebooks.\n\n'
        '## Reproduction and results\n\n'
        'Use an explicit local published dataset root, or explicitly select Hugging Face with the actual published '
        '40-hex commit revision after upload. A local missing file never causes network fallback. No destination '
        'commit is claimed before publication. Executed notebooks and their artifacts are bound in '
        '[notebook-export-manifest.json](notebook-export-manifest.json); portable parameter edits are recorded. '
        'Learned S2–S4 results are **not_run** and require admitted checkpoints and measured result receipts.\n\n'
        f"This card derives its scientific claims from {evidence}.\n\n"
        '## Licensing and citation\n\n'
        'Generator code uses the [MIT license](LICENSE). Derived dataset artifacts retain JAXA ALOS AW3D30 '
        'attribution and upstream usage obligations in [Dataset Terms](DATASET_LICENSE.md). '
        'See [CITATION.cff](CITATION.cff) and [DATASHEET.md](DATASHEET.md).\n')
    citation = yaml.safe_load((source_root / 'CITATION.cff').read_text())
    citation.update(version=VERSION, url=f'https://huggingface.co/datasets/{repo_id}')
    citation.pop('date-released', None)
    citation_text = yaml.safe_dump(citation, sort_keys=False)
    base = _json(source_root / 'croissant.json')
    croissant = {key: base[key] for key in ('@context', 'creator', 'citeAs', 'license', 'conformsTo')}
    croissant.update({'@id': f'https://huggingface.co/datasets/{repo_id}', '@type': 'sc:Dataset',
        'name': 'PASSAGE', 'version': VERSION, 'description': description,
        'url': f'https://huggingface.co/datasets/{repo_id}',
        'distribution': [], 'recordSet': []})
    for name in facts['configurations']:
        sample = name.endswith('_samples')
        fileset_id = f'{name}-parquet'
        croissant['distribution'].append({'@id': fileset_id, '@type': 'cr:FileSet', 'name': name,
            'encodingFormat': 'application/x-parquet', 'includes': f'{name}/*/*.parquet'})
        fields = []
        for column in _features(sample, list(facts['graphs'])):
            field = {'@id': f'{name}/{column}', '@type': 'cr:Field', 'name': column,
                'description': f'Parquet column {column}; exact Arrow feature schema is declared in the README dataset_info.',
                'source': {'fileSet': {'@id': fileset_id}, 'extract': {'column': column}}}
            if column in ('id', 'metadata_json', 'scene_receipt'):
                field['dataType'] = 'sc:Text'
            elif column.startswith('image_'):
                field['dataType'] = 'sc:ImageObject'
            fields.append(field)
        croissant['recordSet'].append({'@id': name, '@type': 'cr:RecordSet', 'name': name, 'field': fields})
    default = context.configured_default if context.configured_default in facts['configurations'] else next(
        name for name in facts['configurations'] if name.endswith('_samples'))
    header = _generate_huggingface_yaml_header(repo_id, facts['configurations'], list(SPLITS), default, list(facts['graphs']))
    outputs = {'README.md': (header + body).encode(), 'DATASHEET.md': (
        '# PASSAGE v1.0 datasheet\n\n' + description + '\n\n'
        'This datasheet documents the computational annotation and qualification process, not model results or '
        'an operational deployment certification.\n\n' + body).encode(),
        'croissant.json': (json.dumps(croissant, indent=2, sort_keys=True) + '\n').encode(),
        'CITATION.cff': citation_text.encode(), FACTS: (json.dumps(facts, indent=2, sort_keys=True) + '\n').encode(),
        'calibration/calibrate.json': (root / 'release-receipts/calibration.json').read_bytes()}
    for name in ('LICENSE', 'DATASET_LICENSE.md'):
        outputs[name] = (source_root / name).read_bytes()
    for relative, content in outputs.items():
        write_staged_bytes(root / relative, content)
    receipt = {'schema_version': 'passage-successor-publication-v1', 'dataset_version': VERSION,
        'repo_id': repo_id, 'status': 'prepared', 'scope': facts['scope'],
        'input_sha256': inputs, 'files_sha256': {relative: _hash(root / relative) for relative in outputs},
        'facts_sha256': _hash(root / FACTS), 'learned_results': 'not_run', 'upload': 'not_run'}
    write_staged_bytes(root / RECEIPT, (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    return verify_successor_publication_metadata(root)


def verify_successor_publication_metadata(root: str | Path) -> dict:
    """Recheck metadata bytes and their scientific source identities before readiness."""
    root = Path(root).resolve()
    receipt = _json(root / RECEIPT)
    if receipt.get('schema_version') != 'passage-successor-publication-v1' or receipt.get('dataset_version') != VERSION:
        raise ValueError('Release publication metadata receipt version mismatch')
    for relative, expected in {**receipt['input_sha256'], **receipt['files_sha256']}.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _hash(path) != expected:
            raise ValueError(f'Release publication metadata hash mismatch: {relative}')
    facts, inputs = _facts(root)
    if inputs != receipt['input_sha256'] or _json(root / FACTS) != facts:
        raise ValueError('Release publication scientific facts differ from verified receipts')
    from passage.export import _validate_croissant, _validate_readme_card
    for valid, message in (_validate_croissant(root / 'croissant.json'),
                           _validate_readme_card(root / 'README.md', set(facts['configurations']))):
        if not valid:
            raise ValueError(f'Invalid release publication metadata: {message}')
    croissant = _json(root / 'croissant.json')
    citation = yaml.safe_load((root / 'CITATION.cff').read_text())
    if croissant.get('version') != VERSION or citation.get('version') != VERSION:
        raise ValueError('Release publication version differs across metadata files')
    expected_range = f"{facts['calibration']['min_m']!r} m to {facts['calibration']['max_m']!r} m"
    if (croissant.get('description') != _description(facts)
            or {record['name'] for record in croissant['recordSet']} != set(facts['configurations'])):
        raise ValueError('Release publication scientific metadata differs from verified facts')
    for filename in ('README.md', 'DATASHEET.md'):
        content = (root / filename).read_text()
        if (expected_range not in content or _description(facts) not in content
                or json.dumps(facts['graphs'], indent=2, sort_keys=True) not in content
                or json.dumps(facts['geographic_policy'], indent=2, sort_keys=True) not in content):
            raise ValueError('Release publication scientific metadata differs from verified facts')
    return {**receipt, 'status': 'verified', 'receipt_sha256': _hash(root / RECEIPT)}
