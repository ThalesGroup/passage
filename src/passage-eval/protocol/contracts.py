"""Serializable ordinary-input, graph, budget and terminal-result contracts.

Only the released eight-neighbour, destination-only obstacle graph is supported.
Reference labels and qualification records belong to the offline evaluator.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from passage_eval.protocol.planner import Scene

STATUSES = frozenset({'ok', 'no_path', 'invalid_path', 'budget_exhausted', 'timeout',
                      'oom', 'runtime_error', 'cancelled'})
STAGES = frozenset({'primary', 'repair', 'fallback'})

# Deliberately finite schemas: an ordinary-looking container must not carry
# nested label/target fields. Calibration extrema describe terrain, not targets.
_METADATA_SCALARS = frozenset({'split', 'resolution', 'index', 'source_id', 'source_revision',
                               'cluster_id', 'task_variant', 'tile_id', 'source_footprint_id', 'source_tile_id'})
_METADATA_CONTAINERS = {
    'calibration': frozenset({'min', 'max', 'elevation_min', 'elevation_max'}),
    'geographic': frozenset({'latitude', 'longitude', 'tile_id', 'cluster_id'}),
}


def ordinary_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy just declared ordinary scalar fields, including nested schemas."""
    if not isinstance(metadata, Mapping):
        raise ValueError('metadata must be a mapping')
    def scalar(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int, float)) or value is None:
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        raise ValueError('ordinary metadata fields must be scalars')
    result = {key: scalar(metadata[key]) for key in _METADATA_SCALARS if key in metadata}
    for key, allowed in _METADATA_CONTAINERS.items():
        if key in metadata:
            if not isinstance(metadata[key], Mapping):
                raise ValueError(f'{key} metadata must be a mapping')
            result[key] = {name: scalar(value) for name, value in metadata[key].items() if name in allowed}
            numeric = allowed if key == 'calibration' else {'latitude', 'longitude'}
            for name in numeric & result[key].keys():
                value = result[key][name]
                if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                    raise ValueError(f'{key}.{name} must be a finite numeric ordinary input')
            if key == 'calibration':
                for low, high in (('min', 'max'), ('elevation_min', 'elevation_max')):
                    if low in result[key] or high in result[key]:
                        if low not in result[key] or high not in result[key] or result[key][high] <= result[key][low]:
                            raise ValueError('calibration requires a complete strictly increasing numeric range')
    return result


@dataclass(frozen=True)
class GraphSpec:
    connectivity: int = 8
    diagonal_policy: str = 'destination_only'
    pixel_size_m: float = 30.0
    obstacle_policy: str = 'blocked'
    coordinate_order: str = 'row_col'
    cost_model: str = 'elevation'
    cost_weight: float = 62.4
    rtol: float = 1e-7
    atol: float = 1e-6

    def __post_init__(self):
        from passage_eval.metrics.cost import canonical_cost_model
        if self.diagonal_policy == 'always':
            object.__setattr__(self, 'diagonal_policy', 'destination_only')
        if (self.connectivity != 8 or self.diagonal_policy != 'destination_only'
                or self.pixel_size_m != 30 or self.obstacle_policy != 'blocked'
                or self.coordinate_order != 'row_col'):
            raise ValueError('only the released 8-connected, 30-m destination-only graph is supported')
        object.__setattr__(self, 'cost_model', canonical_cost_model(self.cost_model))
        for name in ('cost_weight', 'rtol', 'atol'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class BudgetSpec:
    """Cooperative limits, checked again at safe completion.

    A Python call cannot be forcibly stopped. The runner retains ownership until
    it returns, records late/drain time and never admits a late result as success.
    Counters must be reported when a corresponding work budget is configured.
    """
    wall_time_ms: float | None = None
    max_expansions: int | None = None
    max_queue_operations: int | None = None
    max_live_states: int | None = None

    def __post_init__(self):
        if self.wall_time_ms is not None and (isinstance(self.wall_time_ms, (bool, np.bool_))
                or not isinstance(self.wall_time_ms, Real)
                or not np.isfinite(self.wall_time_ms) or self.wall_time_ms <= 0):
            raise ValueError('wall_time_ms must be finite and positive')
        for name in ('max_expansions', 'max_queue_operations', 'max_live_states'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
                raise ValueError(f'{name} must be a positive integer')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PlanRequest:
    sample_id: str
    elevation: np.ndarray
    obstacles: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    cost_model: str
    cost_weight: float
    graph: GraphSpec | None = None
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tensor: np.ndarray | None = None
    deadline_monotonic: float | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        from passage_eval.metrics.cost import canonical_cost_model
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError('sample_id must be a nonempty string')
        model = canonical_cost_model(self.cost_model)
        graph = self.graph or GraphSpec(cost_model=model, cost_weight=self.cost_weight)
        if not isinstance(graph, GraphSpec) or graph.cost_model != model or graph.cost_weight != self.cost_weight:
            raise ValueError('request objective must match GraphSpec')
        if not isinstance(self.budget, BudgetSpec):
            raise ValueError('budget must be a BudgetSpec')
        if self.deadline_monotonic is not None and (isinstance(self.deadline_monotonic, bool)
                or not isinstance(self.deadline_monotonic, Real) or not np.isfinite(self.deadline_monotonic)):
            raise ValueError('deadline_monotonic must be a finite absolute clock value')
        elevation = np.asarray(self.elevation)
        obstacles = np.asarray(self.obstacles)
        if (elevation.ndim != 2 or not elevation.size or not np.isfinite(elevation).all()
                or obstacles.shape != elevation.shape or not np.isin(obstacles, [0, 1]).all()):
            raise ValueError('request grids must be finite 2D terrain and matching binary obstacles')
        for endpoint in (self.start, self.goal):
            values = np.asarray(endpoint)
            if (values.shape != (2,) or not np.isfinite(values).all()
                    or not np.equal(values, np.floor(values)).all()
                    or np.any(values < 0) or np.any(values >= elevation.shape)
                    or obstacles[tuple(values.astype(int))]):
                raise ValueError('request endpoints must be traversable in-bounds integer cells')
        if np.array_equal(self.start, self.goal):
            raise ValueError('the released protocol requires distinct endpoints')
        object.__setattr__(self, 'cost_model', model)
        object.__setattr__(self, 'graph', graph)
        object.__setattr__(self, 'elevation', elevation.copy())
        object.__setattr__(self, 'obstacles', obstacles.copy())
        object.__setattr__(self, 'start', tuple(int(v) for v in self.start))
        object.__setattr__(self, 'goal', tuple(int(v) for v in self.goal))
        object.__setattr__(self, 'metadata', ordinary_metadata(self.metadata))
        if self.input_tensor is not None:
            tensor = np.asarray(self.input_tensor)
            if tensor.dtype != np.float32 or tensor.shape != (*elevation.shape, 3) or not np.isfinite(tensor).all():
                raise ValueError('input_tensor must be a finite float32 (H,W,3) released ordinary input')
            calibration = self.metadata.get('calibration', {})
            if not {'min', 'max'} <= calibration.keys():
                raise ValueError('input_tensor requires finite numeric min/max elevation calibration')
            heights, markers, blocked = (tensor[:, :, index] for index in range(3))
            if np.any(heights < 0) or np.any(heights > 1):
                raise ValueError('input_tensor elevation channel must lie in [0,1]')
            if (not np.isin(markers, [-1, 0, 1]).all() or np.count_nonzero(markers == 0) != 1
                    or np.count_nonzero(markers == 1) != 1 or markers[self.start] != 0 or markers[self.goal] != 1):
                raise ValueError('input_tensor marker channel must encode exactly the declared endpoints')
            if not np.isin(blocked, [0, 1]).all() or not np.array_equal(blocked, obstacles):
                raise ValueError('input_tensor obstacle channel must equal the declared obstacle grid')
            decoded = heights.astype(np.float64) * (calibration['max'] - calibration['min']) + calibration['min']
            if not np.array_equal(decoded, elevation.astype(np.float64)):
                raise ValueError('input_tensor calibrated terrain must equal the declared elevation grid')
            object.__setattr__(self, 'input_tensor', tensor.copy())

    @classmethod
    def from_scene(cls, scene: Scene, *, budget: BudgetSpec | None = None, graph: GraphSpec | None = None):
        return cls(scene.sample_id, scene.elevation, scene.obstacles, scene.start, scene.goal,
                   scene.cost_model, scene.cost_weight, graph=graph, budget=budget or BudgetSpec(),
                   metadata=scene.metadata, input_tensor=scene.input_tensor)

    def candidate_copy(self, *, deadline_monotonic: float | None = None):
        return PlanRequest(self.sample_id, self.elevation, self.obstacles, self.start, self.goal,
                           self.cost_model, self.cost_weight, self.graph, self.budget,
                           self.metadata, self.input_tensor,
                           self.deadline_monotonic if deadline_monotonic is None else deadline_monotonic)

    def remaining_wall_ms(self, *, clock=time.perf_counter) -> float | None:
        """One admission deadline shared by prediction, repair and fallback.

        The runner sets the absolute deadline before copying ordinary inputs.
        Standalone requests without a runner deadline have no absolute limit;
        their caller must initialize one before launching a timed procedure.
        """
        if self.deadline_monotonic is None:
            return None
        return max(0.0, 1000 * (self.deadline_monotonic - clock()))


@dataclass(frozen=True)
class PlanResult:
    """Candidate terminal output; work totals include all primary/repair/fallback.

    stage_work provides subtotals; the runner charges at least their sum, while
    resource peaks use the maximum. precision must describe actual execution.
    Unreported precision remains explicitly unspecified.
    path must own a materialized numeric array or coordinate sequence. The
    runner synchronizes before snapshotting its bytes; lazy iterators are not
    admitted because consuming them can introduce work after that boundary.
    """
    path: Any
    status: str = 'ok'
    stage: str = 'primary'
    work: dict[str, int | float] = field(default_factory=dict)
    stage_work: dict[str, dict[str, int | float]] = field(default_factory=dict)
    precision: str = 'unspecified'
    diagnostics: dict[str, Any] = field(default_factory=dict)
    stage_spans: dict[str, float] = field(default_factory=dict)
    resource_peaks: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f'unknown terminal status {self.status!r}')
        if self.stage not in STAGES or not set(self.stage_work) <= STAGES:
            raise ValueError('stage must be primary, repair or fallback')
        if not isinstance(self.precision, str) or not self.precision:
            raise ValueError('precision must be a nonempty string')

    @property
    def predicted_path(self):
        return self.path
