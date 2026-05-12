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

Sub-commands
============

``bench``      — load val data → run candidate inference → score → submit
``leaderboard``— print the current leaderboard CSV
``rebuild``    — rebuild ``leaderboard.csv`` from the ``runs/`` tree
``init-spec``  — emit a starter ``my_method.passage-bench.yaml``
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from passage_eval import (
    AStarPlanner,
    DEFAULT_SEEDS,
    Planner,
    PlannerOutput,
    Scene,
    Submission,
    __version__,
    apply_censored_penalty,
    load_leaderboard,
    primary_metrics_from_arrays,
    PerSampleArrays,
    rebuild_leaderboard,
    score_planner,
    submit,
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

    # --- benchmark cell --------------------------------------------------
    resolution: int = 256
    cost_model: str = "elevation"
    cost_weight: float = 1.0
    split: str = "val"
    n_samples: int = 1024                  #: cap on val samples to score
    seeds: Sequence[int] = DEFAULT_SEEDS

    # --- deployment metadata --------------------------------------------
    edge_accelerator: str | None = None    #: "Thor-GPU" / "Orin-NPU" / "h100" / None
    batch_size: int = 1
    dtype: str = "fp32"
    measure_latency: bool = True
    latency_warmup: int = 5
    latency_repeats: int = 30

    # --- artefact destinations ------------------------------------------
    results_root: str = "outputs/passage-eval/results"
    dataset_version: str = "thalesgroup/passage:current"
    extra: dict[str, Any] = field(default_factory=dict)

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
# `passage-eval --help-signature`.
inference: my_module:MyPlanner
inference_kwargs:
  checkpoint: outputs/my_method/best.pt

# --- benchmark cell ---------------------------------------------------------
resolution: 256                               # PASSAGE input resolution
cost_model: elevation                         # elevation / slope_uphill / energy / ...
cost_weight: 1.0
split: val
n_samples: 1024                               # cap; the released val split has 1024 entries
seeds: [1234, 2026, 3407]                     # MUST stay aligned with DEFAULT_SEEDS

# --- deployment metadata (purely for leaderboard provenance) ----------------
edge_accelerator: Thor-GPU                    # h100 / Thor-GPU / Orin-GPU / Orin-NPU / Tiger-iGPU / null
batch_size: 1
dtype: fp16
measure_latency: true
latency_warmup: 5
latency_repeats: 30

# --- where to land ----------------------------------------------------------
results_root: outputs/passage-eval/results
dataset_version: thalesgroup/passage:v0.4
extra:
  notes: "first end-to-end submission"
"""


# ---------------------------------------------------------------------------
# Inference entry-point loading
# ---------------------------------------------------------------------------


def _import_entry(spec: str) -> Any:
    """Load ``"module:attr"`` and return the resolved attribute."""
    if ":" not in spec:
        raise ValueError(
            f"inference entry-point must be in 'module:attr' form, got {spec!r}"
        )
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise AttributeError(f"{module_name} has no attribute {attr!r}")
    return getattr(module, attr)


def _materialise_planner(spec: BenchSpec) -> Planner:
    """Resolve ``spec.inference`` into a :class:`Planner` instance.

    Two supported forms:

    * a :class:`Planner` subclass (instantiated with ``inference_kwargs``);
    * a plain callable matching the path-emitting signature, wrapped
      in :class:`_CallablePlanner`.
    """

    obj = _import_entry(spec.inference)
    if isinstance(obj, type) and issubclass(obj, Planner):
        return obj(**spec.inference_kwargs)
    if callable(obj):
        return _CallablePlanner(obj, name=spec.method_name, kwargs=spec.inference_kwargs)
    raise TypeError(f"{spec.inference} is neither a Planner subclass nor callable")


@dataclass
class _CallablePlanner(Planner):
    """Adapter wrapping a free function as a :class:`Planner`."""

    func: Callable[..., Iterable[tuple[int, int]]]
    name: str = "callable"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def predict_path(self, scene: Scene) -> PlannerOutput:
        path = self.func(
            elevation=scene.elevation,
            obstacles=scene.obstacles,
            start=scene.start,
            goal=scene.goal,
            cost_model=scene.cost_model,
            cost_weight=scene.cost_weight,
            **self.kwargs,
        )
        path_list = [tuple(p) for p in path]
        # The metric machinery does not require the path cost; we set 0
        # and let `compute_primary_metrics` re-derive cost-regret from
        # the path mask. Submitters who want the canonical path cost
        # can implement Planner directly instead of the callable form.
        return PlannerOutput(predicted_path=path_list, predicted_cost=0.0)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def _load_scenes(spec: BenchSpec) -> list[Scene]:
    """Load `n_samples` scenes from the released PASSAGE val split.

    Tries the released `passage_eval.io` loader (when wired up); falls
    back to a deterministic synthetic batch so the CLI is exercisable
    even on a clean checkout without dataset access.
    """

    try:
        from passage_eval.io.dataset import load_val_scenes  # noqa: WPS433
        return load_val_scenes(
            resolution=spec.resolution,
            cost_model=spec.cost_model,
            cost_weight=spec.cost_weight,
            n_samples=spec.n_samples,
            split=spec.split,
        )
    except Exception:
        pass

    # Deterministic synthetic fallback — keeps the CLI usable in CI.
    rng = np.random.default_rng(0)
    H = W = spec.resolution
    scenes: list[Scene] = []
    oracle = AStarPlanner()
    for i in range(min(spec.n_samples, 16)):  # synthetic fallback is small
        elevation = rng.uniform(0, 100, size=(H, W)).astype(np.float64)
        obstacles = (rng.uniform(0, 1, size=(H, W)) < 0.05).astype(np.uint8)
        bootstrap = oracle.predict_path(
            Scene(
                sample_id=f"synthetic-{i}",
                elevation=elevation,
                obstacles=obstacles,
                start=(0, 0), goal=(H - 1, W - 1),
                cost_model=spec.cost_model,
                cost_weight=spec.cost_weight,
                oracle_path=[(0, 0)], oracle_cost=0.0,
            )
        )
        scenes.append(
            Scene(
                sample_id=f"synthetic-{i}",
                elevation=elevation,
                obstacles=obstacles,
                start=(0, 0), goal=(H - 1, W - 1),
                cost_model=spec.cost_model,
                cost_weight=spec.cost_weight,
                oracle_path=bootstrap.predicted_path,
                oracle_cost=bootstrap.predicted_cost,
            )
        )
    return scenes


# ---------------------------------------------------------------------------
# bench command
# ---------------------------------------------------------------------------


def cmd_bench(spec: BenchSpec) -> Path:
    """End-to-end benchmark for one candidate planner."""

    print(f"[passage-eval] bench v{__version__}: method={spec.method_name!r} on r{spec.resolution}/{spec.cost_model}")
    planner = _materialise_planner(spec)
    scenes = _load_scenes(spec)
    print(f"[passage-eval] loaded {len(scenes)} scenes; planner = {planner.__class__.__name__}")

    e2e_p50_ms: float | None = None
    if spec.measure_latency:
        e2e_p50_ms = _measure_e2e_latency(
            planner, scenes,
            warmup=spec.latency_warmup,
            repeats=spec.latency_repeats,
        )
        print(f"[passage-eval] latency_e2e_ms_p50 = {e2e_p50_ms:.3f}")

    results_root = Path(spec.results_root).resolve()
    last_run_dir: Path | None = None
    for seed in spec.seeds:
        np.random.seed(int(seed))
        metrics = score_planner(planner, scenes)
        sub = Submission(
            method_name=spec.method_name,
            family=spec.family,
            architecture=spec.architecture,
            decoder_variant=spec.decoder_variant,
            resolution=spec.resolution,
            cost_model=spec.cost_model,
            cost_weight=spec.cost_weight,
            split=spec.split,
            seed=int(seed),
            n_samples=metrics["n_samples"],
            metrics=metrics,
            host=socket.gethostname(),
            edge_accelerator=spec.edge_accelerator,
            batch_size=spec.batch_size,
            dtype=spec.dtype,
            latency_e2e_ms_p50=e2e_p50_ms,
            package_version=__version__,
            dataset_version=spec.dataset_version,
            extra=dict(spec.extra,
                       inference_entry=spec.inference,
                       cli_invocation=" ".join(sys.argv)),
        )
        last_run_dir = submit(sub, root=results_root)
        print(f"[passage-eval] seed {seed}: CR-med={metrics['cost_regret_median']:.4f} "
              f"-> {last_run_dir.relative_to(results_root)}")
    assert last_run_dir is not None
    return last_run_dir


def _measure_e2e_latency(planner: Planner, scenes: list[Scene],
                         *, warmup: int, repeats: int) -> float:
    """Wall-clock end-to-end latency (per-sample p50)."""
    if not scenes:
        return float("nan")
    reference = scenes[0]
    for _ in range(max(0, warmup)):
        planner.predict_path(reference)
    samples_ms: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        planner.predict_path(reference)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples_ms))


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
    p_bench.add_argument("--spec", type=Path, default=None,
                          help="YAML/JSON BenchSpec file (overrides flags)")
    p_bench.add_argument("--method-name", default=None)
    p_bench.add_argument("--family", default=None)
    p_bench.add_argument("--architecture", default=None)
    p_bench.add_argument("--decoder-variant", default="default")
    p_bench.add_argument("--inference", default=None,
                          help="entry-point: 'module:Class' or 'module:callable'")
    p_bench.add_argument("--resolution", type=int, default=256)
    p_bench.add_argument("--cost-model", default="elevation")
    p_bench.add_argument("--cost-weight", type=float, default=1.0)
    p_bench.add_argument("--n-samples", type=int, default=1024)
    p_bench.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p_bench.add_argument("--edge-accelerator", default=None)
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--dtype", default="fp32")
    p_bench.add_argument("--no-latency", action="store_true")
    p_bench.add_argument("--results-root", default="outputs/passage-eval/results")
    p_bench.add_argument("--dataset-version", default="thalesgroup/passage:current")

    p_leader = sub.add_parser("leaderboard", help="print the current leaderboard")
    p_leader.add_argument("--root", default="outputs/passage-eval/results")
    p_leader.add_argument("--metric", default="cost_regret_median")

    p_rebuild = sub.add_parser("rebuild",
                                help="rebuild leaderboard.csv from runs/")
    p_rebuild.add_argument("--root", default="outputs/passage-eval/results")

    p_init = sub.add_parser("init-spec",
                              help="write a starter passage-bench.yaml spec")
    p_init.add_argument("--out", default="my_method.passage-bench.yaml")
    return parser


def _spec_from_args(args: argparse.Namespace) -> BenchSpec:
    if args.spec is not None:
        spec = BenchSpec.from_file(args.spec)
        # CLI overrides
        if args.inference: spec.inference = args.inference
        if args.method_name: spec.method_name = args.method_name
        if args.edge_accelerator: spec.edge_accelerator = args.edge_accelerator
        if args.results_root: spec.results_root = args.results_root
        return spec
    required = ("method_name", "family", "architecture", "inference")
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        raise SystemExit(
            f"`bench` needs --spec FILE or all of {missing!r} via flags"
        )
    return BenchSpec(
        method_name=args.method_name, family=args.family,
        architecture=args.architecture, decoder_variant=args.decoder_variant,
        inference=args.inference,
        resolution=args.resolution, cost_model=args.cost_model,
        cost_weight=args.cost_weight,
        n_samples=args.n_samples, seeds=tuple(args.seeds),
        edge_accelerator=args.edge_accelerator, batch_size=args.batch_size,
        dtype=args.dtype, measure_latency=not args.no_latency,
        results_root=args.results_root, dataset_version=args.dataset_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "bench":
        spec = _spec_from_args(args)
        cmd_bench(spec)
        return 0
    if args.cmd == "leaderboard":
        lb = load_leaderboard(args.root)
        if not len(lb):
            print("(empty leaderboard)"); return 0
        cols = ["method_name", "family", "architecture",
                "edge_accelerator", "seed",
                args.metric, "latency_e2e_ms_p50", "n_samples"]
        df = lb.df[[c for c in cols if c in lb.df.columns]]
        df = df.sort_values(args.metric)
        try: print(df.to_string(index=False))
        except Exception: print(df)
        return 0
    if args.cmd == "rebuild":
        path = rebuild_leaderboard(args.root)
        print(f"rebuilt {path}")
        return 0
    if args.cmd == "init-spec":
        Path(args.out).write_text(PASSAGE_BENCH_SPEC_TEMPLATE)
        print(f"wrote {args.out}")
        return 0
    parser.error(f"unknown command {args.cmd!r}")
    return 1
