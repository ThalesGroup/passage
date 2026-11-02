# MIT License
#
# Copyright (c) 2026 THALES

"""Command-line interface for `passage-eval`.

The CLI's reason for being: a third-party submitter should be able to
take their PASSAGE-compatible inference function, point ``passage-eval
bench`` at it, and get a leaderboard row out — with no glue code, no
manual scoring, and no implicit decisions about which seeds, which
metrics, or which censored-penalty rule are in scope. Everything that
matters is declared up-front via a single :class:`BenchSpec`, either
from CLI flags or from a YAML/JSON spec file.

``conform → run → score → compare → report`` separates candidate execution
from independent reference scoring and matched-task reporting. ``bench --spec``
keeps the legacy submission interface while selecting its primary metrics from
the frozen warm repeat of those same persisted complete requests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socket
import sys
import time
import random
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from passage_eval.protocol.planner import candidate_scene

from passage_eval import (
    DEFAULT_SEEDS,
    Planner,
    PlannerOutput,
    Scene,
    Submission,
    __version__,
    load_leaderboard,
    rebuild_leaderboard,
    submit,
    ReproducibilitySidecar,
    compute_calibration_decisions_hash,
)


__all__ = ["main", "BenchSpec", "PASSAGE_BENCH_SPEC_TEMPLATE"]


# ---------------------------------------------------------------------------
# Public spec
# ---------------------------------------------------------------------------


@dataclass
class BenchSpec:
    """Everything the CLI needs to score one candidate planner end-to-end.

    A submitter typically creates a YAML/JSON copy of this structure
    (see :data:`PASSAGE_BENCH_SPEC_TEMPLATE`), or passes the same
    fields as CLI flags. The dataclass is intentionally small so that
    a reviewer can read the spec in a single screen.
    """

    # --- identity --------------------------------------------------------
    method_name: str
    family: str
    architecture: str
    decoder_variant: str = "default"

    # --- inference entry point ------------------------------------------
    #: ``"my_module:MyPlanner"`` (Planner subclass) **or**
    #: ``"my_module:my_predict_fn"`` (a callable matching :func:`signature_help`).
    inference: str = ""

    #: Optional kwargs forwarded to the planner constructor / callable
    #: factory. JSON-serialisable.
    inference_kwargs: dict[str, Any] = field(default_factory=dict)
    inference_format: str = "tensor"  #: tensor callable, or legacy scene keyword callable
    inference_kwargs_by_seed: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- benchmark cell --------------------------------------------------
    resolution: int = 256
    cost_model: str = "elevation"
    cost_weight: float | None = None
    split: str = "val"
    n_samples: int = 1024  #: cap on val samples to score
    seeds: Sequence[int] = DEFAULT_SEEDS

    # --- deployment metadata --------------------------------------------
    edge_accelerator: str | None = None  #: "Thor-GPU" / "Orin-NPU" / "h100" / None
    batch_size: int = 1
    hardware_stack: str | None = None
    determinism_flag: str = "non_deterministic"
    dtype: str = "fp32"
    measure_latency: bool = True
    latency_warmup: int = 5
    latency_repeats: int | None = None  #: one complete warm repeat unless explicitly declared

    # --- artefact destinations ------------------------------------------
    results_root: str = "outputs/passage-eval/qualified-v1"
    dataset_version: str = "thalesgroup/passage:local-or-pinned"
    dataset_repo: str = "thalesgroup/passage"
    dataset_revision: str | None = None
    dataset_path: str | None = None
    task_variant: str = "obstacles"
    calibration_decisions: dict[str, Any] | None = None
    container_digest: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    reference_max_expansions: int = 10000
    quality_repeat_index: int = 0
    reference_policy: str = "released_labels"
    reference_cohort_id: str | None = None
    runtime_bundle_manifest: str | None = None
    candidate_source_sha256: str | None = None
    component_control: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchSpec":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yml", ".yaml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML spec requires `pyyaml`") from exc
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PASSAGE_BENCH_SPEC_TEMPLATE = """\
# Submission spec for `passage-eval bench` — fill in and run:
#     passage-eval bench --spec my_method.passage-bench.yaml

# --- identity ---------------------------------------------------------------
method_name: my_method                       # short identifier, snake_case
family: rl                                    # dense_regression / dense_segmentation / graph / classical / rl
architecture: ppo_actor_critic                # backbone or class label
decoder_variant: argmax_policy                # post-processing variant

# --- inference entry point --------------------------------------------------
# Either a Planner subclass (`module:Class`) or a callable
# (`module:fn`) matching the signature documented in
# docs/benchmark/contributing-a-method.md.
inference: my_module:MyPlanner
inference_kwargs:
  checkpoint: outputs/my_method/best.pt

# --- benchmark cell ---------------------------------------------------------
resolution: 256                               # PASSAGE input resolution
cost_model: elevation                         # elevation / slope_uphill / energy / ...
cost_weight: null                             # use the released weight for each sample
split: validation
n_samples: 1024                               # deterministic subsample cap
seeds: [1234, 2026, 3407]                     # MUST stay aligned with DEFAULT_SEEDS

# --- deployment metadata (purely for leaderboard provenance) ----------------
edge_accelerator: Thor-GPU                    # h100 / Thor-GPU / Orin-GPU / Orin-NPU / Tiger-iGPU / null
batch_size: 1
dtype: fp16
measure_latency: true
latency_warmup: 5
latency_repeats: null                         # one complete warm repeat by default

# --- where to land ----------------------------------------------------------
results_root: outputs/passage-eval/qualified-v1
dataset_repo: thalesgroup/passage
dataset_revision: null                       # pin a dataset commit for a frozen benchmark
dataset_path: null                           # or a local export root
task_variant: obstacles
inference_format: tensor                     # plan(input_tensor) -> ordered path
inference_kwargs_by_seed: {}                 # seed -> checkpoint kwargs for trained replicas
dataset_version: thalesgroup/passage:local-or-pinned
budget: {}                                  # cooperative wall/work limits
reference_max_expansions: 10000               # bounded offline qualification
quality_repeat_index: 0                      # freeze before execution
reference_policy: released_labels            # or independent_analytic_reference
reference_cohort_id: null                    # required for the independent cohort
runtime_bundle_manifest: null                # optional passage-runtime-bundle-v1 JSON file manifest
candidate_source_sha256: null                # explicit declared identity when no module file exists
extra:
  notes: "first end-to-end submission"
"""


# ---------------------------------------------------------------------------
# Inference entry-point loading
# ---------------------------------------------------------------------------


def _import_entry(spec: str) -> Any:
    """Load ``"module:attr"`` and return the resolved attribute."""
    if ":" not in spec:
        raise ValueError(f"inference entry-point must be in 'module:attr' form, got {spec!r}")
    module_name, attr = spec.split(":", 1)
    # Console-script launchers put their bin directory on sys.path. Support
    # the explicitly requested external module in the caller's working tree,
    # as python -m does, including its local imports during execution.
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise AttributeError(f"{module_name} has no attribute {attr!r}")
    return getattr(module, attr)


def _materialise_planner(spec: BenchSpec) -> Any:
    """Resolve ``spec.inference`` into a :class:`Planner` instance.

    Two supported forms:

    * a :class:`Planner` subclass (instantiated with ``inference_kwargs``);
    * a plain callable matching the path-emitting signature, wrapped
      in :class:`_CallablePlanner`.
    """

    obj = _import_entry(spec.inference)
    if isinstance(obj, type):
        planner = obj(**spec.inference_kwargs)
        if callable(getattr(planner, "plan", None)) or callable(getattr(planner, "predict_path", None)):
            return planner
        raise TypeError(f"{spec.inference} must implement plan(request) or predict_path(scene)")
    if callable(obj):
        if spec.inference_format == "request":
            from functools import partial

            return partial(obj, **spec.inference_kwargs)
        return _CallablePlanner(
            obj, name=spec.method_name, kwargs=spec.inference_kwargs, input_kind=spec.inference_format
        )
    raise TypeError(f"{spec.inference} is neither a Planner subclass nor callable")


@dataclass
class _CallablePlanner(Planner):
    """Adapter wrapping a free function as a :class:`Planner`."""

    func: Callable[..., Iterable[tuple[int, int]]]
    name: str = "callable"
    kwargs: dict[str, Any] = field(default_factory=dict)
    input_kind: str = "scene"

    def predict_path(self, scene: Scene) -> PlannerOutput:
        if self.input_kind == "tensor":
            if scene.input_tensor is None:
                raise ValueError("Tensor callables require the released input_tensor")
            path = self.func(scene.input_tensor, **self.kwargs)
        elif self.input_kind == "scene":
            path = self.func(
                elevation=scene.elevation,
                obstacles=scene.obstacles,
                start=scene.start,
                goal=scene.goal,
                cost_model=scene.cost_model,
                cost_weight=scene.cost_weight,
                **self.kwargs,
            )
        else:
            raise ValueError("inference_format must be tensor or scene")
        # Keep the returned storage owner intact until runner completion sync.
        # Routes must be materialized numeric arrays/sequences; lazy iterators
        # cannot establish their device-work boundary and fail central admission.
        return PlannerOutput(predicted_path=path, predicted_cost=float("nan"))


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def _load_scenes(spec: BenchSpec) -> list[Scene]:
    """Load released scenes; propagate errors rather than substitute synthetic data."""
    from passage_eval.io.dataset import load_val_scenes

    if spec.dataset_path is not None and not Path(spec.dataset_path).is_dir():
        raise FileNotFoundError(f"local dataset root does not exist: {spec.dataset_path}")
    if spec.dataset_path is None and not spec.dataset_revision:
        raise ValueError("remote data requires an explicit dataset_revision")
    return load_val_scenes(
        resolution=spec.resolution,
        cost_model=spec.cost_model,
        cost_weight=spec.cost_weight,
        n_samples=spec.n_samples,
        split=spec.split,
        dataset_path=spec.dataset_path,
        dataset_repo=spec.dataset_repo,
        dataset_revision=spec.dataset_revision,
        task_variant=spec.task_variant,
    )


# ---------------------------------------------------------------------------
# bench command
# ---------------------------------------------------------------------------


def _dataset_content_hash(scenes: list[Scene]) -> str:
    """Hash the evaluated subset, including labels, task weights and input tensors."""
    digest = hashlib.sha256()
    for scene in scenes:
        record = (scene.sample_id, scene.cost_model, scene.cost_weight, scene.start, scene.goal, scene.oracle_cost)
        digest.update(json.dumps(record, sort_keys=True).encode())
        for array in (scene.input_tensor, scene.elevation, scene.obstacles, np.asarray(scene.oracle_path, dtype="<i8")):
            if array is not None:
                value = np.ascontiguousarray(array)
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def _checkout_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hash() -> str:
    """Fingerprint installed generator/evaluator source, including dirty edits."""
    import passage
    import passage_eval

    digest = hashlib.sha256()
    for package in (passage, passage_eval):
        root = Path(package.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            digest.update(f"{package.__name__}/{path.relative_to(root)}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_spec(spec: BenchSpec) -> None:
    from passage_eval.protocol.contracts import BudgetSpec
    from passage_eval.protocol.attribution import component_control

    if not spec.seeds or len(set(spec.seeds)) != len(spec.seeds):
        raise ValueError("seeds must be a nonempty list of distinct values")
    if spec.batch_size != 1:
        raise ValueError("complete-request runner supports batch_size=1 only")
    if spec.latency_warmup < 0:
        raise ValueError("latency_warmup must be nonnegative")
    repeats = spec.latency_repeats if spec.measure_latency and spec.latency_repeats is not None else 1
    if repeats < 1 or not 0 <= spec.quality_repeat_index < repeats:
        raise ValueError("quality_repeat_index must select a positive declared repeat series")
    if spec.reference_max_expansions <= 0:
        raise ValueError("reference_max_expansions must be positive")
    if spec.reference_policy not in {"released_labels", "independent_analytic_reference"}:
        raise ValueError("reference_policy must be released_labels or independent_analytic_reference")
    if spec.reference_policy == "independent_analytic_reference" and (
        not isinstance(spec.reference_cohort_id, str) or not spec.reference_cohort_id.strip()
    ):
        raise ValueError("independent_analytic_reference requires an explicit reference_cohort_id")
    BudgetSpec(**spec.budget)
    component_control(spec.component_control)


def _qualification(scenes: list[Scene], spec: BenchSpec) -> dict[str, Any]:
    from passage.audit import qualified_reference_for_scene

    if spec.reference_policy == "independent_analytic_reference":
        from passage.audit import independently_qualified_reference_for_scene

        records = {
            scene.sample_id: independently_qualified_reference_for_scene(
                scene, cohort_id=spec.reference_cohort_id, max_expansions=spec.reference_max_expansions
            )
            for scene in scenes
        }
    else:
        records = {
            scene.sample_id: qualified_reference_for_scene(scene, max_expansions=spec.reference_max_expansions)
            for scene in scenes
        }
    failed = {
        key: value.get("qualification_reasons", []) for key, value in records.items() if not value.get("qualified")
    }
    if failed:
        raise ValueError(f"reference qualification failed before candidate admission: {failed}")
    return records


def _json_write(path: Path, payload: Any) -> None:
    from passage_eval.execution.artifacts import json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _seed_spec(spec: BenchSpec, seed: int) -> BenchSpec:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(int(seed))
    seed_kwargs = spec.inference_kwargs_by_seed.get(str(seed), spec.inference_kwargs_by_seed.get(int(seed), {}))
    return replace(spec, seeds=[int(seed)], inference_kwargs={**spec.inference_kwargs, **seed_kwargs})


def _candidate_identity(spec: BenchSpec) -> dict[str, Any]:
    """Bind external entry-point bytes and explicit runtime dependencies.

    A source-less entry point needs an explicit identity. Such an identity is
    labelled declared rather than being presented as a measured source hash.
    Runtime bundle files use paths relative to their JSON manifest and verified
    SHA256 values; this captures helpers/native libraries beyond the entry file.
    """
    import inspect
    import marshal

    from passage_eval.execution.artifacts import sha256_file

    obj = _import_entry(spec.inference)
    defining_module = inspect.getmodule(obj)
    requested_module = sys.modules.get(spec.inference.split(":", 1)[0])
    modules = {}
    for module in (requested_module, defining_module):
        path = getattr(module, "__file__", None)
        if path is not None and Path(path).is_file():
            resolved = Path(path).resolve()
            modules[module.__name__] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    loaded_code = {}
    members = (
        {"entry": obj}
        if not isinstance(obj, type)
        else {name: getattr(obj, name, None) for name in ("__init__", "plan", "predict_path")}
    )
    for name, member in members.items():
        code = getattr(member, "__code__", None)
        if code is not None:
            loaded_code[name] = hashlib.sha256(marshal.dumps(code)).hexdigest()
    bundle = None
    if spec.runtime_bundle_manifest is not None:
        path = Path(spec.runtime_bundle_manifest).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "passage-runtime-bundle-v1"
            or not isinstance(payload.get("files"), list)
            or not payload["files"]
        ):
            raise ValueError("runtime bundle manifest requires passage-runtime-bundle-v1 and nonempty files")
        files = {}
        for item in payload["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["path"], str):
                raise ValueError("runtime bundle files require path and sha256")
            member_path = (path.parent / item["path"]).resolve()
            actual = sha256_file(member_path)
            if member_path.as_posix() in files or actual != item["sha256"]:
                raise ValueError("runtime bundle file identity mismatch or duplicate file")
            files[member_path.as_posix()] = actual
        bundle = {"path": str(path), "manifest_sha256": sha256_file(path), "files": files}
    declared = spec.candidate_source_sha256
    if declared is not None and (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ValueError("candidate_source_sha256 must be a lowercase 64-character SHA256 identity")
    defining_name = getattr(defining_module, "__name__", getattr(obj, "__module__", None))
    if defining_name not in modules and bundle is None and declared is None:
        raise ValueError(
            "candidate defining module has no readable file; declare a runtime bundle or source SHA256 identity"
        )
    return {
        "schema": "passage-candidate-identity-v1",
        "entry": spec.inference,
        "defining_module": defining_name,
        "module_files": modules,
        "loaded_code_sha256": loaded_code,
        "runtime_bundle": bundle,
        "declared_source_sha256": declared,
        "source_verification": "file_content_measured" if defining_name in modules else "explicit_declared_identity",
    }


def _checkpoint_hashes(spec: BenchSpec) -> dict[str, str]:
    checkpoint_hashes = {}
    for key, value in spec.inference_kwargs.items():
        if "checkpoint" in key or "bundle" in key:
            if not isinstance(value, str) or not Path(value).is_file():
                raise FileNotFoundError(f"declared {key} must identify a readable local artifact: {value!r}")
            with Path(value).open("rb") as stream:
                checkpoint_hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    return checkpoint_hashes


def _manifest(spec: BenchSpec, scenes: list[Scene], qualification: dict[str, Any]) -> dict[str, Any]:
    from passage_eval.protocol.attribution import component_control

    control = component_control(spec.component_control)
    checkpoint_hashes = _checkpoint_hashes(spec)
    candidate_identity = _candidate_identity(spec)
    procedure = {
        "method_name": spec.method_name,
        "family": spec.family,
        "architecture": spec.architecture,
        "decoder_variant": spec.decoder_variant,
        "inference": spec.inference,
        "inference_format": spec.inference_format,
        "inference_kwargs": spec.inference_kwargs,
        "budget": spec.budget,
        "checkpoint_hashes": checkpoint_hashes,
        "candidate_identity": candidate_identity,
        "component_control": control,
    }
    reference_identity = hashlib.sha256(json.dumps(qualification, sort_keys=True).encode()).hexdigest()
    return {
        "bench_spec": spec.to_dict(),
        "method_name": spec.method_name,
        "seed": int(spec.seeds[0]),
        "procedure": procedure,
        "procedure_id": hashlib.sha256(json.dumps(procedure, sort_keys=True).encode()).hexdigest(),
        "checkpoint_hashes": checkpoint_hashes,
        "runtime_source_hash": _source_hash(),
        "candidate_identity": candidate_identity,
        "component_control": control,
        "reference_qualification": qualification,
        "reference_identity": reference_identity,
        "admission": {"qualified_reference": True, "ordinary_input_contract": "PlanRequest-v1"},
        "reference_policy": spec.reference_policy,
        "reference_cohort_id": spec.reference_cohort_id,
        "dataset": {
            "repo": spec.dataset_repo,
            "revision": spec.dataset_revision,
            "path": str(Path(spec.dataset_path).resolve()) if spec.dataset_path else None,
            "split": spec.split,
            "resolution": spec.resolution,
            "task_variant": spec.task_variant,
            "content_hash": _dataset_content_hash(scenes),
        },
        "split_role": "evaluation",
        "hardware_stack": spec.hardware_stack or f"{socket.gethostname()}-cpu",
        "cold_series_scope": "first complete roster pass after construction; only roster[0] is first request; import/model loading excluded",
        "extra": spec.extra,
    }


def _execute_series(spec: BenchSpec, scenes: list[Scene], qualification: dict[str, Any], output: Path):
    from passage_eval.execution import run_requests
    from passage_eval.protocol.contracts import BudgetSpec, PlanRequest

    manifest = _manifest(spec, scenes, qualification)
    planner = _materialise_planner(spec)
    if _candidate_identity(spec) != manifest["candidate_identity"]:
        raise ValueError("candidate source/runtime identity changed during construction")
    if _checkpoint_hashes(spec) != manifest["checkpoint_hashes"]:
        raise ValueError("checkpoint artifact identity changed during construction")
    requests = [PlanRequest.from_scene(scene, budget=BudgetSpec(**spec.budget)) for scene in scenes]
    output.mkdir(parents=True, exist_ok=True)
    _json_write(output / "qualification.json", qualification)
    artifacts = []
    if spec.measure_latency:
        artifacts.append(
            run_requests(
                planner,
                requests,
                output_dir=output / "cold",
                manifest=manifest,
                synchronize=_sync_runtime,
                series="cold",
            )
        )
        if spec.latency_warmup:
            artifacts.append(
                run_requests(
                    planner,
                    requests,
                    repeats=spec.latency_warmup,
                    output_dir=output / "warmup",
                    manifest=manifest,
                    synchronize=_sync_runtime,
                    series="warmup",
                )
            )
    artifacts.append(
        run_requests(
            planner,
            requests,
            repeats=(spec.latency_repeats or 1) if spec.measure_latency else 1,
            output_dir=output,
            manifest=manifest,
            synchronize=_sync_runtime,
            quality_repeat_index=spec.quality_repeat_index,
            series="warm",
        )
    )
    return artifacts


def cmd_run(spec: BenchSpec, output: str | Path) -> list[Path]:
    """Qualify the entire roster before constructing or calling any candidate."""
    _validate_spec(spec)
    scenes = _load_scenes(spec)
    qualification = _qualification(scenes, spec)
    _import_entry(spec.inference)
    paths = []
    for seed in spec.seeds:
        seed_output = Path(output) / f"seed-{seed}"
        if seed_output.exists() and any(seed_output.iterdir()):
            raise FileExistsError(f"run output already exists: {seed_output}")
        _execute_series(_seed_spec(spec, int(seed)), scenes, qualification, seed_output)
        paths.append(seed_output)
    return paths


def cmd_score(run: str | Path, output: str | Path | None = None) -> list[Path]:
    """Read persisted routes and rescore every series without invoking inference."""
    import shutil
    from passage_eval.execution import read_run_artifact, score_run_artifact, write_scored_artifact

    root = Path(run)
    if output is not None and Path(output).resolve() != root.resolve():
        if Path(output).resolve().is_relative_to(root.resolve()):
            raise ValueError("score output must be outside the source run directory")
        root = Path(shutil.copytree(root, output))
    manifests = sorted(root.rglob("run_manifest.json"))
    if not manifests:
        raise ValueError(f"no complete-request artifacts found in {root}")
    paths = []
    for manifest_path in manifests:
        artifact = read_run_artifact(manifest_path.parent)
        procedure = artifact.manifest.get("procedure")
        candidate_identity = artifact.manifest.get("candidate_identity")
        if (
            not isinstance(procedure, dict)
            or not isinstance(candidate_identity, dict)
            or procedure.get("candidate_identity") != candidate_identity
            or candidate_identity.get("schema") != "passage-candidate-identity-v1"
            or hashlib.sha256(json.dumps(procedure, sort_keys=True).encode()).hexdigest()
            != artifact.manifest.get("procedure_id")
        ):
            raise ValueError("persisted candidate/procedure identity is missing or inconsistent")
        spec = BenchSpec(**artifact.manifest["bench_spec"])
        scenes = _load_scenes(spec)
        if _dataset_content_hash(scenes) != artifact.manifest["dataset"]["content_hash"]:
            raise ValueError("dataset content differs from the admitted run")
        metrics, arrays = score_run_artifact(artifact, scenes)
        write_scored_artifact(artifact, metrics, arrays)
        paths.append(manifest_path.parent)
    return paths


def cmd_conform(spec: BenchSpec, output: str | Path) -> Path:
    """Exercise the shared analytic graph and a bounded local candidate slice."""
    from passage_eval.baselines.classical import BoundedExactPlanner
    from passage_eval.execution import run_requests, score_run_artifact, write_scored_artifact
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.execution.conformance import harness_conformance

    _validate_spec(spec)
    scene = Scene(
        "analytic-3x3",
        np.zeros((3, 3)),
        np.zeros((3, 3), dtype=bool),
        (0, 0),
        (2, 2),
        "elevation",
        62.4,
        [(0, 0), (1, 1), (2, 2)],
        60 * np.sqrt(2),
    )
    fixture_spec = replace(
        spec,
        n_samples=1,
        seeds=[spec.seeds[0]],
        latency_warmup=0,
        latency_repeats=1,
        quality_repeat_index=0,
        measure_latency=False,
    )
    qualified = _qualification(
        [scene], replace(fixture_spec, reference_policy="released_labels", reference_cohort_id=None)
    )
    artifact = run_requests(
        BoundedExactPlanner(),
        [PlanRequest.from_scene(scene)],
        output_dir=Path(output) / "analytic",
        manifest={"reference_qualification": qualified},
    )
    metrics, arrays = score_run_artifact(artifact, [scene])
    write_scored_artifact(artifact, metrics, arrays)
    if metrics["success_rate"] != 1 or not np.isclose(metrics["cost_regret_median"], 0):
        raise ValueError("analytic graph conformance failed")
    checks = harness_conformance(scene)
    paths = cmd_run(fixture_spec, Path(output) / "candidate")
    cmd_score(Path(output) / "candidate")
    receipt = Path(output) / "conformance.json"
    _json_write(
        receipt,
        {
            "schema": "passage-conformance-v1",
            "status": "passed",
            "analytic_fixture": str(artifact.output_dir),
            "candidate_runs": [str(path) for path in paths],
            "harness_checks": checks,
            "claim_scope": "cpu_contract_checks_not_device_or_population_certification",
            "scope": "analytic graph and one qualified ordinary-input request",
            "candidate_quality_requirement": "terminal outcomes retained; no optimality threshold",
        },
    )
    return receipt


def cmd_bench(spec: BenchSpec) -> Path:
    """Compatibility leaderboard row selected from timed, persisted requests."""
    import shutil
    import tempfile
    from passage_eval.execution import score_run_artifact, write_scored_artifact
    from passage_eval.metrics.summary import aggregate_primary_metrics

    _validate_spec(spec)
    scenes = _load_scenes(spec)
    qualification = _qualification(scenes, spec)
    _import_entry(spec.inference)
    results_root = Path(spec.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    last_run_dir = None
    for seed in spec.seeds:
        seed_spec = _seed_spec(spec, int(seed))
        with tempfile.TemporaryDirectory(prefix="complete-", dir=results_root) as temporary:
            artifacts = _execute_series(seed_spec, scenes, qualification, Path(temporary))
            for artifact in artifacts:
                scored_metrics, scored = score_run_artifact(artifact, scenes)
                write_scored_artifact(artifact, scored_metrics, scored)
            warm = artifacts[-1]
            _, arrays = score_run_artifact(warm, scenes)
            mask = arrays["repeat_index"] == spec.quality_repeat_index
            primary = {
                key: value[mask] if value.ndim and len(value) == len(mask) else value for key, value in arrays.items()
            }
            metrics = aggregate_primary_metrics(primary)
            latencies = np.asarray([record["total_wall_ms"] for record in warm.records])
            timing = {
                "latency_e2e_ms_p50": float(np.percentile(latencies, 50)),
                "latency_e2e_ms_p95": float(np.percentile(latencies, 95)),
                "latency_e2e_ms_p99": float(np.percentile(latencies, 99)),
                "throughput_samples_per_s": 1000 / float(latencies.mean()),
                "throughput_kind": "serial_complete_request_service_capacity",
                "latency_repeats": warm.manifest["repeats"],
                "latency_warmup": spec.latency_warmup,
                "latency_population": len(warm.records),
                "quality_repeat_index": spec.quality_repeat_index,
            }
            provenance = dict(
                spec.extra,
                **timing,
                inference_entry=spec.inference,
                inference_kwargs=seed_spec.inference_kwargs,
                inference_format=spec.inference_format,
                dataset_repo=spec.dataset_repo,
                dataset_revision=spec.dataset_revision,
                dataset_path=spec.dataset_path,
                task_variant=spec.task_variant,
                metric_contract=str(primary["metric_contract"].item()),
                evaluator_source_hash=_source_hash(),
                evaluation_batch_size=1,
                seed_set=list(spec.seeds),
                sample_ids=[scene.sample_id for scene in scenes],
                reference_identity=warm.manifest["reference_identity"],
                procedure_id=warm.manifest["procedure_id"],
                budget=warm.manifest["procedure"]["budget"],
                candidate_identity=warm.manifest["candidate_identity"],
                component_control=warm.manifest["component_control"],
                checkpoint_hashes=warm.manifest["checkpoint_hashes"],
                dataset_content_hash=warm.manifest["dataset"]["content_hash"],
                hardware_stack=spec.hardware_stack or spec.edge_accelerator or f"{socket.gethostname()}-cpu",
                determinism_flag=spec.determinism_flag,
                container_digest=spec.container_digest,
                calibration_decisions_hash=compute_calibration_decisions_hash(spec.calibration_decisions)
                if spec.calibration_decisions is not None
                else "",
                reference_policy=spec.reference_policy,
                reference_cohort_id=spec.reference_cohort_id,
            )
            sidecar = ReproducibilitySidecar(
                method_name=spec.method_name,
                family=spec.family,
                dataset_version=spec.dataset_version,
                dataset_content_hash=_dataset_content_hash(scenes),
                operating_point=f"{spec.resolution}x{spec.resolution}/{scenes[0].cost_model}",
                seeds=[int(seed)],
                split_seed=42,
                hardware_stack=spec.hardware_stack or spec.edge_accelerator or f"{socket.gethostname()}-cpu",
                determinism_flag=spec.determinism_flag,
                batch_size=1,
                dtype=spec.dtype,
                commit_sha=_checkout_commit(),
                container_digest=spec.container_digest,
                calibration_decisions_hash=compute_calibration_decisions_hash(spec.calibration_decisions)
                if spec.calibration_decisions is not None
                else "",
                extra=provenance,
                notes="Primary metrics use the frozen warm repeat; all measured series are independently scored.",
            )
            weights = [scene.cost_weight for scene in scenes]
            submission = Submission(
                method_name=spec.method_name,
                family=spec.family,
                architecture=spec.architecture,
                decoder_variant=spec.decoder_variant,
                resolution=spec.resolution,
                cost_model=scenes[0].cost_model,
                cost_weight=weights[0] if min(weights) == max(weights) else float("nan"),
                split=spec.split,
                seed=int(seed),
                n_samples=metrics["n_samples"],
                metrics=metrics,
                host=socket.gethostname(),
                edge_accelerator=spec.edge_accelerator,
                batch_size=1,
                dtype=spec.dtype,
                latency_e2e_ms_p50=timing["latency_e2e_ms_p50"] if spec.measure_latency else None,
                package_version=__version__,
                dataset_version=spec.dataset_version,
                extra=provenance,
            )
            last_run_dir = submit(submission, root=results_root, per_sample=primary, sidecar=sidecar)
            shutil.copytree(temporary, last_run_dir / "complete_requests")
            print(f"[passage-eval] seed {seed}: persisted timed requests -> {last_run_dir}")
    return last_run_dir


def _sync_runtime() -> None:
    """Complete queued CUDA work when a candidate imported PyTorch."""
    runtime = sys.modules.get("torch")
    if runtime is not None and runtime.cuda.is_available():
        runtime.cuda.synchronize()


def _measure_e2e_latencies(planner: Planner, scenes: list[Scene], *, warmup: int, repeats: int) -> dict[str, float]:
    """Deprecated timing-only helper, retained for caller compatibility.

    Its discarded outputs are not admissible benchmark evidence. Workflow and
    bench commands use the complete-request runner instead.
    """
    if not scenes or repeats <= 0 or warmup < 0:
        raise ValueError("timing requires scenes, positive repeats and nonnegative warmup")
    for i in range(warmup):
        planner.predict_path(candidate_scene(scenes[i % len(scenes)]))
    samples_ms = []
    for i in range(repeats):
        current_input = candidate_scene(scenes[i % len(scenes)])
        _sync_runtime()
        t0 = time.perf_counter()
        planner.predict_path(current_input)
        _sync_runtime()
        samples_ms.append((time.perf_counter() - t0) * 1000)
    p50, p95, p99 = np.percentile(samples_ms, [50, 95, 99])
    return {
        "latency_e2e_ms_p50": float(p50),
        "latency_e2e_ms_p95": float(p95),
        "latency_e2e_ms_p99": float(p99),
        "throughput_samples_per_s": 1000 / float(np.mean(samples_ms)),
        "latency_repeats": repeats,
        "latency_warmup": warmup,
    }


def _measure_e2e_latency(planner: Planner, scenes: list[Scene], *, warmup: int, repeats: int) -> float:
    """Compatibility wrapper returning the median input-to-path latency."""
    return _measure_e2e_latencies(planner, scenes, warmup=warmup, repeats=repeats)["latency_e2e_ms_p50"]


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passage-eval",
        description="Canonical benchmarking CLI for the PASSAGE evaluation contract.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("bench", help="score a candidate planner end-to-end")
    p_bench.add_argument(
        "--spec", type=Path, default=None, help="YAML/JSON BenchSpec file; explicit flags override its fields"
    )
    p_bench.add_argument("--method-name", default=None)
    p_bench.add_argument("--family", default=None)
    p_bench.add_argument("--architecture", default=None)
    p_bench.add_argument("--decoder-variant", default=None)
    p_bench.add_argument("--inference", default=None, help="entry-point: 'module:Class' or 'module:callable'")
    p_bench.add_argument("--resolution", type=int, default=None)
    p_bench.add_argument("--cost-model", default=None)
    p_bench.add_argument(
        "--cost-weight",
        type=float,
        default=None,
        help="confirm the released per-sample weight; never redefine the oracle",
    )
    p_bench.add_argument("--n-samples", type=int, default=None)
    p_bench.add_argument("--seeds", type=int, nargs="+", default=None)
    p_bench.add_argument("--edge-accelerator", default=None)
    p_bench.add_argument("--batch-size", type=int, default=None)
    p_bench.add_argument("--dtype", default=None)
    p_bench.add_argument("--no-latency", action="store_true")
    p_bench.add_argument("--results-root", default=None)
    p_bench.add_argument("--dataset-path", default=None, help="local export root containing <N>x<N>/<split> shards")
    p_bench.add_argument("--dataset-repo", default=None)
    p_bench.add_argument("--dataset-revision", default=None)
    p_bench.add_argument("--split", choices=["val", "validation", "calibration", "test"], default=None)
    p_bench.add_argument("--task-variant", choices=["free", "obstacles"], default=None)
    p_bench.add_argument("--inference-format", choices=["request", "tensor", "scene"], default=None)
    p_bench.add_argument("--latency-warmup", type=int, default=None)
    p_bench.add_argument("--latency-repeats", type=int, default=None)
    p_bench.add_argument("--dataset-version", default=None)
    p_bench.add_argument("--quality-repeat-index", type=int, default=None)
    p_bench.add_argument("--reference-max-expansions", type=int, default=None)
    p_bench.add_argument("--reference-policy", choices=["released_labels", "independent_analytic_reference"])
    p_bench.add_argument("--reference-cohort-id")
    p_bench.add_argument("--runtime-bundle-manifest")
    p_bench.add_argument("--candidate-source-sha256")

    for command in ("conform", "run"):
        workflow = sub.add_parser(command, help=f"{command} a declared ordinary-input procedure")
        workflow.add_argument("--spec", required=True, type=Path, help="JSON/YAML BenchSpec")
        workflow.add_argument("--output", required=True, type=Path)

    score = sub.add_parser("score", help="independently rescore every persisted timed route")
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--output", type=Path, help="copy run artifacts here before scoring")

    compare = sub.add_parser("compare", help="compare matched, qualified scored artifacts")
    compare.add_argument("--score", required=True, action="append", type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--primary-repeat-index", type=int)
    compare.add_argument("--bootstrap-resamples", type=int, default=2000)
    compare.add_argument("--bootstrap-seed", type=int, default=0)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--odd-spec", type=Path, help="frozen JSON axes and bin edges for joint ODD strata")

    report = sub.add_parser("report", help="render a persisted comparison")
    report.add_argument("--compare", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--format", choices=["markdown", "json"], default="markdown")

    p_leader = sub.add_parser("leaderboard", help="print the current leaderboard")
    p_leader.add_argument("--root", default="outputs/passage-eval/qualified-v1")
    p_leader.add_argument("--metric", default="cost_regret_median")

    p_rebuild = sub.add_parser("rebuild", help="rebuild leaderboard.csv from runs/")
    p_rebuild.add_argument("--root", default="outputs/passage-eval/qualified-v1")

    p_init = sub.add_parser("init-spec", help="write a starter passage-bench.yaml spec")
    p_init.add_argument("--out", default="my_method.passage-bench.yaml")
    return parser


def _spec_from_args(args: argparse.Namespace) -> BenchSpec:
    if args.spec is not None:
        spec = BenchSpec.from_file(args.spec)
    else:
        required = ("method_name", "family", "architecture", "inference")
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"`bench` needs --spec FILE or all of {missing!r} via flags")
        spec = BenchSpec(args.method_name, args.family, args.architecture, inference=args.inference)
    for key in spec.to_dict():
        value = getattr(args, key, None)
        if value is not None:
            setattr(spec, key, value)
    if args.no_latency:
        spec.measure_latency = False
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd in {"conform", "run"}:
        spec = BenchSpec.from_file(args.spec)
        result = cmd_conform(spec, args.output) if args.cmd == "conform" else cmd_run(spec, args.output)
        print(result)
        return 0
    if args.cmd == "score":
        print(cmd_score(args.run, args.output))
        return 0
    if args.cmd == "compare":
        from passage_eval.metrics.comparison import compare_artifacts

        result = compare_artifacts(
            args.score,
            primary_repeat_index=args.primary_repeat_index,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
            odd_spec=json.loads(args.odd_spec.read_text()) if args.odd_spec else None,
        )
        _json_write(args.output, result)
        print(args.output)
        return 0
    if args.cmd == "report":
        from passage_eval.metrics.comparison import report_artifact

        result = report_artifact(args.compare, format=args.format)
        if args.format == "json":
            _json_write(args.output, result)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
        print(args.output)
        return 0
    if args.cmd == "bench":
        spec = _spec_from_args(args)
        cmd_bench(spec)
        return 0
    if args.cmd == "leaderboard":
        lb = load_leaderboard(args.root)
        if not len(lb):
            print("(empty leaderboard)")
            return 0
        cols = [
            "method_name",
            "family",
            "architecture",
            "edge_accelerator",
            "seed",
            args.metric,
            "latency_e2e_ms_p50",
            "n_samples",
        ]
        df = lb.df[[c for c in cols if c in lb.df.columns]]
        df = df.sort_values(args.metric)
        try:
            print(df.to_string(index=False))
        except Exception:
            print(df)
        return 0
    if args.cmd == "rebuild":
        path = rebuild_leaderboard(args.root)
        print(f"rebuilt {path}")
        return 0
    if args.cmd == "init-spec":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(PASSAGE_BENCH_SPEC_TEMPLATE)
        print(f"wrote {args.out}")
        return 0
    parser.error(f"unknown command {args.cmd!r}")
    return 1
