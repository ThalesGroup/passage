"""Validate score semantics against the frozen qualified reference payload.

Checksums establish file agreement, not that values are scientifically true.
Reference rows are bound to the admission payload and canonical solver receipt;
geometry is recomputed from the exact persisted candidate and reference routes.
"""
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from passage_eval.execution.artifacts import sha256_file
from passage_eval.metrics.geometric import path_assd, path_hausdorff_p95
from passage_eval.protocol.planner import validated_raw_path


def validate_score_references(artifact, arrays, receipt, directory, indices):
    manifest = artifact.manifest
    reference_file = directory / 'score_reference_inputs.json'
    if 'reference_inputs_sha256' in receipt:
        if not reference_file.is_file() or sha256_file(reference_file) != receipt['reference_inputs_sha256']:
            raise ValueError('qualified reference input artifact hash mismatch')
        bundle = json.loads(reference_file.read_text())
        if (bundle.get('schema') != 'passage-qualified-score-inputs-v1'
                or bundle.get('request_specs') != manifest['request_specs']):
            raise ValueError('qualified reference input task binding mismatch')
        references = bundle.get('reference_qualification')
    else:
        # Older current-contract CLI artifacts already froze the complete
        # reference payload at admission. Missing external payloads fail closed.
        references = manifest.get('reference_qualification')
    if not isinstance(references, Mapping) or set(references) != set(manifest['roster']):
        raise ValueError('comparison requires the complete frozen qualified reference payload')
    identity = hashlib.sha256(json.dumps(references, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if identity != manifest.get('reference_identity'):
        raise ValueError('qualified reference payload differs from the frozen reference identity')
    if receipt.get('reference_payload_identity', identity) != identity:
        raise ValueError('score receipt qualified reference identity mismatch')
    if 'reference_qualification' in manifest and manifest['reference_qualification'] != references:
        raise ValueError('score references disagree with the admitted qualification store')
    routes = {}
    for sample_id, reference in references.items():
        spec = manifest['request_specs'][sample_id]
        exact = reference.get('reference') if isinstance(reference, Mapping) else None
        if (reference.get('qualified') is not True or reference.get('reference_status') != 'exact'
                or not isinstance(exact, Mapping) or exact.get('contract_id') != 'passage-analytic-reference-v1'):
            raise ValueError('comparison requires a qualified canonical exact reference receipt')
        payload = {key: value for key, value in exact.items() if key not in {'reference_id', 'hash'}}
        reference_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        optimum = reference.get('optimal_cost')
        path = validated_raw_path(reference.get('optimal_path', []))
        if (reference_id != reference.get('reference_id') or reference_id != exact.get('reference_id')
                or reference_id != exact.get('hash') or exact.get('status') != 'exact'
                or exact.get('scene_id') != reference.get('scene_id') or not reference.get('scene_id')
                or exact.get('graph_id') != spec['graph_id'] or reference.get('graph_id') != spec['graph_id']
                or exact.get('graph') != spec['graph'] or reference.get('cost_model') != spec['cost_model']
                or reference.get('cost_weight') != spec['cost_weight'] or reference.get('task_variant') != spec['task_variant']
                or not isinstance(optimum, (int, float)) or isinstance(optimum, bool) or not np.isfinite(optimum) or optimum <= 0
                or exact.get('cost') != optimum or not np.array_equal(exact.get('path'), path)
                or path.shape[0] < 2 or not np.array_equal(path[0], spec['start'])
                or not np.array_equal(path[-1], spec['goal']) or np.any(path < 0) or np.any(path >= spec['shape'])
                or not np.all(np.max(np.abs(np.diff(path, axis=0)), axis=1) == 1)):
            raise ValueError('qualified reference receipt/task/cost/path identity mismatch')
        policy = reference.get('qualification_policy', 'released_labels')
        if policy == 'released_labels':
            if (manifest.get('reference_policy', 'released_labels') != 'released_labels'
                    or any(reference.get(key) is not True for key in ('legal', 'stored_cost_agreement', 'optimality'))):
                raise ValueError('released reference qualification policy mismatch')
        elif policy == 'independent_analytic_reference_v1':
            if (manifest.get('reference_policy') != 'independent_analytic_reference'
                    or not reference.get('cohort_id') or reference['cohort_id'] != manifest.get('reference_cohort_id')):
                raise ValueError('independent reference cohort admission mismatch')
        else:
            raise ValueError('unknown reference qualification policy')
        routes[sample_id] = path
        selected = arrays['sample_id'] == sample_id
        if (not np.all(arrays['reference_id'][selected] == reference_id)
                or not np.all(arrays['oracle_costs'][selected] == optimum)):
            raise ValueError('scored reference IDs/costs disagree with the frozen qualified payload')
    for row, raw in zip(artifact.records, artifact.paths):
        index = indices[(row['sample_id'], row['repeat_index'], row['series'])]
        if arrays['valid_path'][index]:
            candidate = validated_raw_path(raw)
            reference = routes[row['sample_id']]
            assd = path_assd([candidate], [reference])[0]
            h95 = path_hausdorff_p95([candidate], [reference])[0]
            if (not np.isclose(arrays['assd'][index], assd, rtol=1e-12, atol=1e-12)
                    or not np.isclose(arrays['h95'][index], h95, rtol=1e-12, atol=1e-12)):
                raise ValueError('scored geometry disagrees with persisted route and qualified reference')
        elif not np.isnan(arrays['assd'][index]) or not np.isnan(arrays['h95'][index]):
            raise ValueError('failed-route geometry must remain explicitly censored')
