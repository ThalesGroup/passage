"""Versioned evaluator-only label/reference and geographic footprint auditing.

Bounded exact solves never certify censored tasks. Stored labels are retained
separately from independent analytic graph references. Geographic source-tile
support, actual crop intersection and source-cell overlap are distinct evidence.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import numpy as np

from passage.dataset import DatasetSource, SPLITS, decode_scene_inputs, normalize_cell
from passage.pathfinding_utils import PIXEL_SIZE_M
from passage_eval.metrics.cost import canonical_cost_model, path_cost
from passage_eval.metrics.reliability import path_status
from passage_eval.protocol.contracts import GraphSpec

AUDIT_CONTRACT = 'passage-audit-v1'
REFERENCE_CONTRACT = 'passage-analytic-reference-v1'


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stored_label_identity(stored_path, stored_cost, *, task_variant: str) -> str:
    """Bind ordered source coordinates and stored cost to their task variant.

    Nonfinite or malformed labels remain hashable audit findings. This identity
    belongs only to evaluator evidence; it never enters ordinary solver inputs.
    """
    def canonical(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            return canonical(value.tolist())
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, float) and not math.isfinite(value):
            return {'nonfinite': str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return {'invalid_type': type(value).__name__, 'value': repr(value)}
    try:
        points = np.asarray(stored_path, dtype=np.float64)
        path = {'shape': list(points.shape), 'coordinates': canonical(points)}
    except (TypeError, ValueError):
        path = {'malformed': canonical(stored_path)}
    try:
        cost = canonical(float(stored_cost))
    except (TypeError, ValueError):
        cost = {'malformed': canonical(stored_cost)}
    return _hash({'contract_id': 'passage-stored-label-identity-v1',
                  'task_variant': task_variant, 'path': path, 'cost': cost})


def scene_identity(elevation, obstacles=None, start=None, goal=None, graph: GraphSpec | None = None) -> str:
    """Bind decoded ordinary inputs and objective, independently of stored labels."""
    if obstacles is None:
        scene = elevation
        elevation, obstacles, start, goal = scene.elevation, scene.obstacles, scene.start, scene.goal
        graph = graph or GraphSpec(cost_model=scene.cost_model, cost_weight=scene.cost_weight)
    if graph is None:
        raise ValueError('scene identity requires an objective GraphSpec')
    digest = hashlib.sha256()
    for array in (np.asarray(elevation, dtype='<f8'), np.asarray(obstacles, dtype=np.uint8)):
        digest.update(str(array.shape).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    digest.update(json.dumps({'start': list(map(int, start)), 'goal': list(map(int, goal)),
                              'graph': graph.to_dict()}, sort_keys=True).encode())
    return digest.hexdigest()


def exact_reference(elevation, obstacles, start, goal, *, cost_model: str, weight: float,
                    max_expansions: int = 10000, graph: GraphSpec | None = None) -> dict[str, Any]:
    """A bounded deterministic A* solve under the canonical positive directed graph.

    Euclidean 30 m distance is an admissible/consistent lower bound for every
    supported objective. Obstacles block destination cells; corner cutting is
    allowed by the release. Budget exhaustion yields no claimed optimum.
    """
    model = canonical_cost_model(cost_model)
    graph = graph or GraphSpec(cost_model=model, cost_weight=weight)
    if graph.cost_model != model or graph.cost_weight != weight:
        raise ValueError('reference objective differs from GraphSpec')
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, Integral) or max_expansions < 0:
        raise ValueError('max_expansions must be a nonnegative integer')
    elevation = np.asarray(elevation, dtype=np.float64)
    obstacles = np.asarray(obstacles)
    if elevation.ndim != 2 or not elevation.size or not np.isfinite(elevation).all():
        raise ValueError('reference elevation must be a finite nonempty 2D grid')
    if obstacles.shape != elevation.shape or not np.isin(obstacles, [0, 1]).all():
        raise ValueError('reference obstacles must match the terrain and be binary')
    endpoints = []
    for point in (start, goal):
        if len(point) != 2 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in point):
            raise ValueError('reference endpoints must be integer row/column cells')
        point = tuple(map(int, point))
        if any(x < 0 or x >= size for x, size in zip(point, elevation.shape)):
            raise ValueError('reference endpoint outside terrain')
        endpoints.append(point)
    start, goal = endpoints
    if start == goal:
        raise ValueError('reference endpoints must be distinct')
    identity = scene_identity(elevation, obstacles, start, goal, graph)
    expansions = 0
    queue_operations = 0
    max_live_states = 0

    def receipt(status, cost=None, path=None):
        result = {'contract_id': REFERENCE_CONTRACT, 'status': status, 'cost': cost, 'path': path or [],
                  'expansions': expansions, 'queue_operations': queue_operations,
                  'max_live_states': max_live_states, 'max_expansions': int(max_expansions),
                  'scene_id': identity, 'graph_id': graph.identity(), 'graph': graph.to_dict()}
        result['reference_id'] = _hash(result)
        result['hash'] = result['reference_id']
        return result

    if obstacles[start] or obstacles[goal]:
        return receipt('no_path')
    elevation_min, elevation_max = float(elevation.min()), float(elevation.max())
    elevation_range = elevation_max - elevation_min
    absolute_max = max(abs(elevation_min), abs(elevation_max))

    def edge(src, dst):
        distance = PIXEL_SIZE_M * math.hypot(dst[0] - src[0], dst[1] - src[1])
        if model == 'distance':
            return distance
        if model in {'elevation', 'absolute_elevation'}:
            if model == 'absolute_elevation':
                value = abs(float(elevation[dst])) / absolute_max if absolute_max > 0 else 0.0
            else:
                value = (float(elevation[dst]) - elevation_min) / elevation_range if elevation_range > 0 else 0.0
            return distance * (1 + weight * value)
        gain = max(0.0, float(elevation[dst] - elevation[src]))
        if model == 'energy':
            return distance + weight * gain
        return distance * (1 + weight * gain / elevation_range) if elevation_range > 0 else distance

    def heuristic(cell):
        return PIXEL_SIZE_M * math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    distances = {start: 0.0}
    previous = {}
    closed = set()
    queue = [(heuristic(start), 0.0, start)]
    queue_operations = 1
    max_live_states = 1
    while queue:
        _, cost, cell = heapq.heappop(queue)
        queue_operations += 1
        if cell in closed or cost != distances[cell]:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            # Re-cost using the independent public metric boundary.
            exact_cost = path_cost(elevation, path, weight=weight, cost_model=model)
            return receipt('exact', float(exact_cost), [list(point) for point in path])
        if expansions >= max_expansions:
            return receipt('budget_exhausted')
        closed.add(cell)
        expansions += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                neighbor = (cell[0] + di, cell[1] + dj)
                if (neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] >= elevation.shape[0]
                        or neighbor[1] >= elevation.shape[1] or obstacles[neighbor] or neighbor in closed):
                    continue
                candidate = cost + edge(cell, neighbor)
                if candidate < distances.get(neighbor, float('inf')):
                    distances[neighbor] = candidate
                    previous[neighbor] = cell
                    heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
                    queue_operations += 1
                    max_live_states = max(max_live_states, len(distances))
    return receipt('no_path')


def _qualified_record(*, sample_id, resolution, split, elevation, obstacles, start, goal,
                      cost_model, weight, task_variant, stored_path, stored_cost,
                      max_expansions, graph=None) -> dict[str, Any]:
    graph = graph or GraphSpec(cost_model=cost_model, cost_weight=weight)
    label_identity = stored_label_identity(stored_path, stored_cost, task_variant=task_variant)
    connected, violated, traversed = path_status(stored_path, start=start, goal=goal, obstacles=obstacles)
    legal = connected and not violated
    recost = path_cost(elevation, stored_path, weight=weight, cost_model=cost_model)
    recost = float(recost) if np.isfinite(recost) else None
    try:
        stored_cost = float(stored_cost)
    except (TypeError, ValueError):
        stored_cost = None
    if stored_cost is not None and (not np.isfinite(stored_cost) or stored_cost <= 0):
        stored_cost = None
    agreement = recost is not None and stored_cost is not None and bool(np.isclose(
        recost, stored_cost, rtol=graph.rtol, atol=graph.atol))
    reference = exact_reference(elevation, obstacles, start, goal, cost_model=cost_model, weight=weight,
                                max_expansions=max_expansions, graph=graph)
    optimality = (bool(np.isclose(recost, reference['cost'], rtol=graph.rtol, atol=graph.atol))
                  if legal and recost is not None and reference['status'] == 'exact' else None)
    reasons = []
    if not legal:
        reasons.append('stored_path_illegal')
    if not agreement:
        reasons.append('stored_cost_disagreement' if stored_cost is not None else 'stored_cost_missing_or_invalid')
    if reference['status'] != 'exact':
        reasons.append('reference_' + reference['status'])
    elif optimality is not True:
        reasons.append('stored_path_suboptimal' if legal else 'stored_path_optimum_unassessable')
    return {'sample_id': sample_id, 'resolution': resolution, 'split': split,
            'cost_model': canonical_cost_model(cost_model), 'cost_weight': float(weight), 'task_variant': task_variant,
            'stored_path': [list(point) for point in stored_path] if legal else None,
            'stored_cost': stored_cost, 'stored_label_id': label_identity, 'recost': recost, 'legal': bool(legal),
            'connected': connected, 'obstacle_violation': violated, 'obstacle_cells_traversed': traversed,
            'stored_cost_agreement': agreement, 'reference_status': reference['status'],
            'optimal_cost': reference['cost'], 'optimal_path': reference['path'], 'optimality': optimality,
            'qualified': not reasons, 'qualification_reasons': reasons,
            'reference_id': reference['reference_id'], 'graph_id': reference['graph_id'],
            'scene_id': reference['scene_id'], 'graph': graph.to_dict(),
            'reference_expansions': reference['expansions'], 'reference': reference}


def audit_row(row: Mapping[str, Any], *, resolution: int, split: str, max_expansions: int = 10000,
              cost_models: Iterable[str] | None = None,
              task_variants: Iterable[str] = ('free', 'obstacles')) -> dict[str, Any]:
    """Retain one finding per declared solver/variant, including malformed labels."""
    resolution, split = normalize_cell(resolution, split)
    variants = tuple(task_variants)
    if not variants or any(variant not in {'free', 'obstacles'} for variant in variants):
        raise ValueError('task_variants must contain free/obstacles')
    requested = {canonical_cost_model(model) for model in cost_models} if cost_models is not None else None
    result = {'sample_id': str(row.get('id', '<missing-id>')), 'resolution': resolution, 'split': split,
              'status': 'audited', 'records': [], 'footprint': geographic_footprint(row)}
    try:
        inputs = decode_scene_inputs(row, resolution=resolution, split=split)
        solvers = inputs['metadata']['solvers']
        if not isinstance(solvers, Mapping) or not solvers:
            raise ValueError('missing released solver metadata')
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status='invalid_metadata', error=str(exc))
        return result
    seen = set()
    model_counts = Counter()
    for cfg in solvers.values():
        try:
            model_counts[canonical_cost_model(cfg['cost_model'])] += 1
        except (KeyError, TypeError, ValueError):
            pass
    for solver_name, cfg in solvers.items():
        try:
            model = canonical_cost_model(cfg['cost_model'])
            if requested is not None and model not in requested:
                continue
            if model_counts[model] != 1:
                raise ValueError(f'multiple solvers for {model}')
            seen.add(model)
            weight = float(cfg['weight'])
            if cfg['diagonal_movement'] != 'always':
                raise ValueError('released connectivity differs from destination-only diagonals')
            graph = GraphSpec(cost_model=model, cost_weight=weight)
        except (ValueError, KeyError, TypeError) as exc:
            result['status'] = 'invalid_metadata'
            for variant in variants:
                result['records'].append({'sample_id': inputs['sample_id'], 'resolution': resolution, 'split': split,
                                          'solver_name': solver_name, 'task_variant': variant, 'qualified': False,
                                          'qualification_reasons': ['invalid_solver_metadata'], 'error': str(exc),
                                          'reference_status': 'not_run', 'legal': None, 'stored_cost_agreement': None})
            continue
        for variant in variants:
            obstacles = inputs['obstacles'] if variant == 'obstacles' else np.zeros_like(inputs['obstacles'])
            stored_path = row.get(f'path_{solver_name}_{variant}', [])
            try:
                stored_cost = inputs['metadata']['paths'][solver_name][variant]['cost']
            except (KeyError, TypeError):
                stored_cost = None
            record = _qualified_record(sample_id=inputs['sample_id'], resolution=resolution, split=split,
                                       elevation=inputs['elevation'], obstacles=obstacles, start=inputs['start'],
                                       goal=inputs['goal'], cost_model=model, weight=weight, task_variant=variant,
                                       stored_path=stored_path, stored_cost=stored_cost,
                                       max_expansions=max_expansions, graph=graph)
            record['solver_name'] = solver_name
            result['records'].append(record)
    if requested is not None and requested - seen:
        result['status'] = 'invalid_metadata'
        result['error'] = f'Missing requested solver models: {sorted(requested - seen)}'
        for model in sorted(requested - seen):
            if model_counts[model]:
                continue
            for variant in variants:
                result['records'].append({'sample_id': inputs['sample_id'], 'resolution': resolution, 'split': split,
                    'cost_model': model, 'task_variant': variant, 'qualified': False,
                    'qualification_reasons': ['missing_solver_metadata'], 'reference_status': 'not_run',
                    'legal': None, 'stored_cost_agreement': None})
    return result


def qualified_reference_for_scene(scene, *, max_expansions: int = 10000) -> dict[str, Any]:
    """Independently qualify a legacy Scene without exposing labels to a request."""
    metadata = scene.metadata or {}
    variant = metadata.get('task_variant')
    if variant is None:
        # A legacy Scene may lack variant metadata. Keep this explicit rather
        # than claiming it is interchangeable with a released obstacle task.
        variant = 'obstacles'
    return _qualified_record(sample_id=scene.sample_id, resolution=scene.elevation.shape[0],
                             split=metadata.get('split', 'unknown'), elevation=scene.elevation,
                             obstacles=scene.obstacles, start=scene.start, goal=scene.goal,
                             cost_model=scene.cost_model, weight=scene.cost_weight, task_variant=variant,
                             stored_path=scene.oracle_path, stored_cost=scene.oracle_cost,
                             max_expansions=max_expansions)


def independently_qualified_reference_for_scene(scene, *, cohort_id: str,
                                                max_expansions: int = 10000) -> dict[str, Any]:
    """Admit a named fresh-reference diagnostic cohort, retaining label defects.

    This does not qualify the released stored segmentation/regression labels.
    Only the independently computed analytic optimum supplies offline scoring
    for this explicitly distinct cohort; source labels remain audit findings.
    """
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError('fresh-reference admission requires an explicit named cohort')
    record = qualified_reference_for_scene(scene, max_expansions=max_expansions)
    record['source_label_qualified'] = record['qualified']
    record['source_label_qualification_reasons'] = record['qualification_reasons'][:]
    record['qualification_policy'] = 'independent_analytic_reference_v1'
    record['cohort_id'] = cohort_id
    exact = record['reference_status'] == 'exact'
    if exact:
        connected, violation, _ = path_status(record['optimal_path'], start=scene.start, goal=scene.goal,
                                              obstacles=scene.obstacles)
        recost = path_cost(scene.elevation, record['optimal_path'], weight=scene.cost_weight,
                           cost_model=scene.cost_model)
        graph = GraphSpec(cost_model=scene.cost_model, cost_weight=scene.cost_weight)
        exact = connected and not violation and bool(np.isclose(recost, record['optimal_cost'],
                                                                 rtol=graph.rtol, atol=graph.atol))
    record['qualified'] = exact
    record['qualification_reasons'] = ([] if exact else ['independent_reference_' + record['reference_status']])
    return record


def reference_store(receipt: Mapping[str, Any], cost_model: str, task_variant: str,
                    qualified_only: bool = True) -> dict[str, dict[str, Any]]:
    """Select an evaluator-only exact-reference mapping for one benchmark cell."""
    model = canonical_cost_model(cost_model)
    store = {}
    for record in receipt['records']:
        if record.get('cost_model') != model or record['task_variant'] != task_variant:
            continue
        if qualified_only and not record['qualified']:
            continue
        key = record['sample_id']
        if key in store:
            raise ValueError(f'Duplicate reference task ID {key}')
        store[key] = record
    return store


def audit_source(source: DatasetSource, *, resolution: int, split: str, limit: int = 1,
                 max_expansions: int = 10000, cost_models: Iterable[str] | None = None,
                 task_variants: Iterable[str] = ('free', 'obstacles'), samples: bool = False) -> dict[str, Any]:
    """Produce a bounded receipt, never describe a pilot as a complete census."""
    if limit is None:
        raise ValueError('audit_source requires an explicit finite limit')
    resolution, split = normalize_cell(resolution, split)
    cost_models = tuple(cost_models) if cost_models is not None else None
    task_variants = tuple(task_variants)
    findings = []
    identities = set()
    records = []
    for row in source.iter_rows(resolution, split, samples=samples, limit=limit):
        sample_id = row.get('id')
        if isinstance(sample_id, str) and sample_id.strip():
            if sample_id in identities:
                raise ValueError(f'Duplicate audit task ID {sample_id}')
            identities.add(sample_id)
        finding = audit_row(row, resolution=resolution, split=split, max_expansions=max_expansions,
                            cost_models=cost_models, task_variants=task_variants)
        findings.append(finding)
        records.extend(finding['records'])
    reasons = Counter(reason for record in records for reason in record['qualification_reasons'])
    header_census = source.split_manifest(resolution, samples=samples)
    receipt = {'contract_id': AUDIT_CONTRACT, 'source': source.identity(), 'resolution': resolution, 'split': split,
               'configuration': f'{resolution}x{resolution}' + ('_samples' if samples else ''),
               'population': 'preview' if samples else 'core', 'max_expansions': max_expansions,
               'coverage': {'requested_limit': limit, 'audited_rows': len(findings), 'task_variants_audited': len(records),
                            'population_total': header_census[split]['count'], 'complete_population': False},
               'counts': {'invalid_metadata': sum(f['status'] == 'invalid_metadata' for f in findings),
                          'legal': sum(r.get('legal') is True for r in records),
                          'stored_cost_agreement': sum(r.get('stored_cost_agreement') is True for r in records),
                          'reference_exact': sum(r['reference_status'] == 'exact' for r in records),
                          'reference_censored': sum(r['reference_status'] == 'budget_exhausted' for r in records),
                          'reference_no_path': sum(r['reference_status'] == 'no_path' for r in records),
                          'qualified': sum(r['qualified'] for r in records), 'qualification_reasons': dict(reasons)},
               'header_census': header_census, 'rows': findings, 'records': records, 'task_ids': sorted(identities),
               'qualified_task_ids': sorted(f['sample_id'] for f in findings if f['status'] == 'audited'
                   and f['records'] and all(record['qualified'] for record in f['records'])),
               'qualification_rule': 'legal AND stored_cost_agreement AND independently_exact_optimal'}
    receipt['receipt_id'] = _hash(receipt)
    return receipt


def geographic_footprint(row: Mapping[str, Any]) -> dict[str, Any]:
    """Use only recorded crop coordinates; missing cell windows remain unknown."""
    metadata = row.get('metadata') or {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    result = {'sample_id': row.get('id'), 'split': SPLITS.get(metadata.get('split'), 'unknown') if isinstance(metadata.get('split'), str) else 'unknown',
              'status': 'unknown',
              'boxes': [], 'source_tiles': [], 'source_cell_status': 'unknown', 'source_cells': []}
    tile = metadata.get('tile') or {}
    tile = tile if isinstance(tile, Mapping) else {}
    overlapping = tile.get('overlapping')
    if isinstance(overlapping, list):
        for item in overlapping:
            if isinstance(item, Mapping):
                name = item.get('name')
                if name:
                    result['source_tiles'].append(str(name))
    # Optional explicit source windows are supported for future releases. They
    # must be identified pixel windows, not source/used tile lists.
    windows = metadata.get('source_cell_windows')
    if isinstance(windows, list) and windows:
        try:
            for window in windows:
                bounds = window['bounds']
                if (not isinstance(window['source_id'], str) or not window['source_id'].strip() or len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, Integral)
                        for v in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
                    raise ValueError('invalid cell window')
                result['source_cells'].append({'source_id': window['source_id'], 'bounds': list(bounds)})
            result['source_cell_status'] = 'known'
        except (KeyError, ValueError, TypeError):
            result['source_cells'] = []
            result['source_cell_status'] = 'invalid'
    try:
        crop = metadata['crop']
        north, west = float(crop['top_left']['lat']), float(crop['top_left']['lon'])
        south, east = float(crop['bottom_right']['lat']), float(crop['bottom_right']['lon'])
        if (not np.isfinite([south, north, west, east]).all() or not -90 <= south < north <= 90
                or not -180 <= west <= 180 or not -180 <= east <= 180 or west == east):
            raise ValueError('invalid crop bounds')
        boxes = [[west, south, east, north]] if west < east else [
            [west, south, 180.0, north], [-180.0, south, east, north]]
        result.update(status='known', boxes=boxes, coordinate_system='recorded_lat_lon_degrees',
                      precision='released_metadata_precision')
    except (KeyError, TypeError, ValueError) as exc:
        result['reason'] = str(exc)
    return result


def _intersection(left, right):
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    return width > 0 and height > 0, width >= 0 and height >= 0


def _geographic_intersection(left, right):
    positive, touch = _intersection(left, right)
    if not touch and (left[0] == -180 and right[2] == 180 or left[2] == 180 and right[0] == -180):
        touch = min(left[3], right[3]) >= max(left[1], right[1])
    return positive, touch


def footprint_overlaps(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Independent cross-split crop, boundary, tile-support and cell checks.

    This bounded pairwise implementation is for finite pilots. Full inventories
    require a separately declared spatial index/census producer.
    """
    footprints = [geographic_footprint(row) for row in rows]
    result = {'contract_id': AUDIT_CONTRACT, 'audited_rows': len(footprints),
              'geographic_coverage': dict(Counter(f['status'] for f in footprints)),
              'source_cell_coverage': dict(Counter(f['source_cell_status'] for f in footprints)),
              'cross_split_geographic_overlaps': [], 'cross_split_boundary_intersections': [],
              'cross_split_shared_source_tiles': [], 'cross_split_source_cell_overlaps': [],
              'footprints': footprints, 'complete_population': False}
    for index, left in enumerate(footprints):
        for right in footprints[index + 1:]:
            if left['split'] == right['split'] or 'unknown' in (left['split'], right['split']):
                continue
            pair = {'task_ids': [left['sample_id'], right['sample_id']], 'splits': [left['split'], right['split']]}
            geographic = [_geographic_intersection(a, b) for a in left['boxes'] for b in right['boxes']]
            if any(positive for positive, _ in geographic):
                result['cross_split_geographic_overlaps'].append(pair)
            elif any(touch for _, touch in geographic):
                result['cross_split_boundary_intersections'].append(pair)
            shared = sorted(set(left['source_tiles']) & set(right['source_tiles']))
            if shared:
                result['cross_split_shared_source_tiles'].append({**pair, 'source_tiles': shared})
            cells = [(a, b) for a in left['source_cells'] for b in right['source_cells']
                     if a['source_id'] == b['source_id'] and _intersection(a['bounds'], b['bounds'])[0]]
            if cells:
                result['cross_split_source_cell_overlaps'].append({**pair,
                    'source_ids': sorted({a['source_id'] for a, _ in cells})})
    result['receipt_id'] = _hash(result)
    return result
