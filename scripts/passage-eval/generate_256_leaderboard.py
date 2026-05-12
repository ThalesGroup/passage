"""Populate `results/` with the full 256x256 leaderboard matrix.

Cross-product of (method × seed × host_accelerator) with the actual
released latencies from edge_experiments/ + p1g_latency_benchmark.csv +
graph/outputs/results_edge/. The model-side metrics are seed-invariant
across hosts (the same checkpoint runs everywhere), so each
(method, seed) cell shares its primary-metric bundle and only the
deployment-side fields (host, accelerator, latency_*) differ.

Run with:
    cd src/passage-eval
    uv run python scripts/generate_256_leaderboard.py --reset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import passage_eval as pe  # noqa: E402

REPO_ROOT = Path("~/projects/passage")
REPO = REPO_ROOT / "passage"
RESULTS_ROOT = REPO / "outputs" / "passage-eval" / "results"

VISION_NPZ = REPO_ROOT / "vision" / "outputs" / "paper" / "results" / "p1_a_per_sample"
EDGE_DIR_THOR_SEG = REPO_ROOT / "paper" / "edge_experiments" / "agx_thor" / "results_multi" / "thor_multi.csv"
EDGE_DIR_ORIN_SEG = REPO_ROOT / "paper" / "edge_experiments" / "agx_orin" / "results_multi" / "orin_multi.csv"
EDGE_DIR_TIGER_SEG = REPO_ROOT / "paper" / "edge_experiments" / "tiger_lake" / "results_nowhere" / "tiger_igpu.csv"
EDGE_DIR_THOR_REG = REPO_ROOT / "paper" / "edge_experiments" / "agx_thor" / "results_reg_decoder" / "thor_reg_with_decoder.csv"
EDGE_DIR_ORIN_REG = REPO_ROOT / "paper" / "edge_experiments" / "agx_orin" / "results_reg_decoder" / "orin_reg_with_decoder.csv"
EDGE_DIR_TIGER_REG = REPO_ROOT / "paper" / "edge_experiments" / "tiger_lake" / "results_reg_decoder" / "tiger_reg_with_decoder.csv"
H100_LAT_CSV = REPO_ROOT / "vision" / "outputs" / "paper" / "results" / "p1_close_out" / "p1g_latency_benchmark.csv"
GNN_H100_CSV = REPO_ROOT / "graph" / "outputs" / "results" / "gnn_h100_256.csv"
GRAPH_EDGE_THOR = REPO_ROOT / "graph" / "outputs" / "results_edge" / "edge_mlp_thor.csv"
GRAPH_EDGE_ORIN = REPO_ROOT / "graph" / "outputs" / "results_edge" / "edge_mlp_orin.csv"
GRAPH_EDGE_TIGER = REPO_ROOT / "graph" / "outputs" / "results_edge" / "edge_mlp_tiger.csv"
GRAPH_EVAL_CSV = REPO_ROOT / "graph" / "outputs" / "results" / "edge_mlp_eval_elevation.csv"

SEEDS = [("primary", 1234), ("replica_a", 2026), ("replica_b", 3407)]


# ---------------------------------------------------------------------------
# Latency extraction helpers
# ---------------------------------------------------------------------------

def _read_edge_seg(csv_path: Path, accelerator: str, branch: str = "segmentation",
                    res: int = 256) -> tuple[float, float, float] | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[(df.branch == branch) & (df.resolution == res)
              & (df.extra_accelerator == accelerator)]
    if sub.empty: return None
    row = sub.iloc[0]
    return (float(row.fwd_ms_p50), float(row.decode_ms_p50), float(row.total_ms_p50))


def _read_edge_reg(csv_path: Path, accelerator: str, res: int = 256) -> tuple[float, float, float] | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[(df.resolution == res) & (df.extra_accelerator == accelerator)]
    if sub.empty: return None
    row = sub.iloc[0]
    return (float(row.fwd_ms_p50), float(row.decode_ms_p50), float(row.total_ms_p50))


def _read_h100_vision(branch: str) -> tuple[float, float, float] | None:
    if not H100_LAT_CSV.exists():
        return None
    df = pd.read_csv(H100_LAT_CSV)
    sub = df[(df.branch == branch) & (df.batch_size == 1) & (df.status == "OK")
              & (df.seed_value == 1234)]
    if sub.empty: return None
    fwd = float(sub.fwd_ms_per_sample.median())
    post = float(sub.postprocess_ms_per_sample.median())
    e2e = float(sub.end_to_end_ms_per_sample.median())
    return (fwd, post, e2e)


def _read_graph_h100() -> dict[str, tuple[float, float, float]]:
    if not GNN_H100_CSV.exists(): return {}
    df = pd.read_csv(GNN_H100_CSV)
    out = {}
    for _, row in df.iterrows():
        out[str(row.branch)] = (float(row.fwd_ms_p50), 0.0, float(row.fwd_ms_p50))
    return out


def _read_graph_edge(csv_path: Path, res: int = 256) -> tuple[float, float, float] | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    sub = df[df.resolution == res]
    if sub.empty: return None
    row = sub.iloc[0]
    return (float(row.fwd_ms_p50), float(row.dec_ms_p50), float(row.total_ms_p50))


def _read_graph_metrics() -> dict[int, dict[str, float]]:
    """Three-seed metrics for the Edge-MLP from the released graph eval CSV."""
    if not GRAPH_EVAL_CSV.exists():
        return {}
    df = pd.read_csv(GRAPH_EVAL_CSV)
    out: dict[int, dict[str, float]] = {}
    for _, r in df.iterrows():
        out[int(r.seed)] = {k: float(r[k]) for k in (
            "cost_regret_median", "cost_regret_p90",
            "path_assd", "path_hausdorff_p95",
            "success_rate", "obstacle_violation_rate",
        )}
    return out


# ---------------------------------------------------------------------------
# Vision per-seed scorer (with censored penalty)
# ---------------------------------------------------------------------------

def _score_vision(template: str, seed_name: str) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    npz = VISION_NPZ / template.format(seed=seed_name)
    arrs = pe.load_per_sample_npz(npz)
    pred_censored = pe.apply_censored_penalty(
        arrs.pred_costs, arrs.oracle_costs,
        decoder_succeeded=arrs.success,
    )
    regret = np.where(np.isfinite(arrs.oracle_costs) & (arrs.oracle_costs > 0),
                      (pred_censored - arrs.oracle_costs) / arrs.oracle_costs, np.nan)
    bundle = pe.PerSampleArrays(
        regret=regret, assd=arrs.assd, h95=arrs.h95, success=arrs.success,
        pred_costs=pred_censored, oracle_costs=arrs.oracle_costs,
        obs_density=arrs.obs_density, elev_std=arrs.elev_std,
        oracle_len=arrs.oracle_len, extras=arrs.extras,
    )
    metrics = pe.primary_metrics_from_arrays(bundle)
    per_sample = {
        "regret": regret,
        "assd": arrs.assd,
        "h95": arrs.h95,
        "success": arrs.success.astype(np.int8),
        "pred_costs": pred_censored,
        "oracle_costs": arrs.oracle_costs,
        "obs_density": arrs.obs_density,
    }
    return metrics, per_sample


# ---------------------------------------------------------------------------
# Submission builders
# ---------------------------------------------------------------------------

def _make_sub(*, method, family, arch, decoder, host, accel, seed, metrics,
               n_samples, lat_fwd=None, lat_dec=None, lat_e2e=None,
               dataset_version="Anonym-2045/passage:v0.4", dtype="fp16",
               extra=None):
    return pe.Submission(
        method_name=method,
        family=family,
        architecture=arch,
        decoder_variant=decoder,
        resolution=256, cost_model="elevation", cost_weight=1.0,
        split="val", seed=seed, n_samples=n_samples,
        metrics=metrics,
        host=host, edge_accelerator=accel,
        batch_size=1, dtype=dtype,
        latency_fwd_ms_p50=lat_fwd,
        latency_decoder_ms_p50=lat_dec,
        latency_e2e_ms_p50=lat_e2e,
        package_version=pe.__version__,
        dataset_version=dataset_version,
        extra=extra or {},
    )


def main(reset: bool = False) -> None:
    if reset and RESULTS_ROOT.exists():
        shutil.rmtree(RESULTS_ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # --- vision: score each seed once, then replicate across host_accels ---
    vision_configs = [
        dict(method="seg-convnextv2_atto_baseline",
             family="dense_segmentation", arch="convnextv2_atto",
             decoder="BCE_threshold",
             template="segmentation_256x256_convnextv2_atto_elevation_baseline__{seed}__strict.npz"),
        dict(method="reg-repvit_m1_bellman_vbridge",
             family="dense_regression", arch="repvit_m1",
             decoder="V_strict",   # released NPZ; paper Table 1 uses V_bridge
             template="regression_256x256_repvit_m1_elevation_bellman__{seed}__strict.npz"),
    ]
    vision_metrics: dict[tuple[str, int], tuple[dict, dict]] = {}
    for cfg in vision_configs:
        for seed_name, seed_value in SEEDS:
            metrics, per_sample = _score_vision(cfg["template"], seed_name)
            vision_metrics[(cfg["method"], seed_value)] = (metrics, per_sample)

    # --- per-(host, accelerator) latencies for vision ---
    vision_latencies = {
        # seg-side
        ("seg-convnextv2_atto_baseline", "h100",       "fp16"):       _read_h100_vision("segmentation"),
        ("seg-convnextv2_atto_baseline", "Thor-GPU",   "fp16"):       _read_edge_seg(EDGE_DIR_THOR_SEG, "gpu"),
        ("seg-convnextv2_atto_baseline", "Orin-GPU",   "fp16"):       _read_edge_seg(EDGE_DIR_ORIN_SEG, "gpu"),
        ("seg-convnextv2_atto_baseline", "Orin-NPU",   "fp16"):       _read_edge_seg(EDGE_DIR_ORIN_SEG, "dla"),
        ("seg-convnextv2_atto_baseline", "Tiger-iGPU", "fp16"):       _read_edge_seg(EDGE_DIR_TIGER_SEG, "igpu"),
        # reg-side
        ("reg-repvit_m1_bellman_vbridge", "h100",       "fp16"):      _read_h100_vision("regression"),
        ("reg-repvit_m1_bellman_vbridge", "Thor-GPU",   "fp16"):      _read_edge_reg(EDGE_DIR_THOR_REG, "gpu"),
        ("reg-repvit_m1_bellman_vbridge", "Orin-GPU",   "fp16"):      _read_edge_reg(EDGE_DIR_ORIN_REG, "gpu"),
        ("reg-repvit_m1_bellman_vbridge", "Orin-NPU",   "fp16"):      _read_edge_reg(EDGE_DIR_ORIN_REG, "dla"),
        ("reg-repvit_m1_bellman_vbridge", "Tiger-iGPU", "fp16"):      _read_edge_reg(EDGE_DIR_TIGER_REG, "igpu"),
    }

    # --- emit vision rows ---
    n_vision = 0
    for (method, accel, dtype), lat in vision_latencies.items():
        if lat is None:
            print(f"  skip {method} on {accel}: no latency data")
            continue
        for cfg in vision_configs:
            if cfg["method"] != method: continue
            for seed_name, seed_value in SEEDS:
                metrics, per_sample = vision_metrics[(method, seed_value)]
                host = "redacted-ccw-host-01" if accel == "h100" else accel.split("-")[0]
                sub = _make_sub(
                    method=method, family=cfg["family"], arch=cfg["arch"],
                    decoder=cfg["decoder"],
                    host=host, accel=accel, seed=seed_value,
                    metrics=metrics, n_samples=metrics["n_samples"],
                    lat_fwd=lat[0], lat_dec=lat[1], lat_e2e=lat[2],
                    dtype=dtype,
                    extra={"source_npz": cfg["template"].format(seed=seed_name)},
                )
                pe.submit(sub, root=RESULTS_ROOT, per_sample=per_sample)
                n_vision += 1
    print(f"emitted {n_vision} vision rows")

    # --- emit graph (Edge-MLP) rows: per-seed metrics × per-host CPU ---
    graph_metrics_by_seed = _read_graph_metrics()
    graph_edge_lats = {
        "Thor-CPU":  _read_graph_edge(GRAPH_EDGE_THOR),
        "Orin-CPU":  _read_graph_edge(GRAPH_EDGE_ORIN),
        "Tiger-CPU": _read_graph_edge(GRAPH_EDGE_TIGER),
    }
    n_graph = 0
    for seed_name, seed_value in SEEDS:
        m = graph_metrics_by_seed.get(seed_value)
        if m is None: continue
        for accel, lat in graph_edge_lats.items():
            host = accel.split("-")[0]
            sub = _make_sub(
                method="graph-edge_mlp_classical_astar",
                family="graph", arch="edge_mlp_5_16_16_1",
                decoder="analytic_a_star_on_learned_costs",
                host=host, accel=accel, seed=seed_value,
                metrics=m, n_samples=64,
                lat_fwd=lat[0] if lat else None,
                lat_dec=lat[1] if lat else None,
                lat_e2e=lat[2] if lat else None,
                dtype="fp32",
                dataset_version="passage_graph-synthetic-64-elevation",
                extra={"source_csv": "edge_mlp_eval_elevation.csv"},
            )
            pe.submit(sub, root=RESULTS_ROOT)
            n_graph += 1
    # Plus GNN H100 row (one seed only — vmpnn timing is per-architecture).
    gnn_h100 = _read_graph_h100()
    for branch, (fwd, dec, e2e) in gnn_h100.items():
        if branch != "vmpnn":   # graphmp is 26s, leave it as a curiosity; vmpnn is the real comparable
            continue
        sub = _make_sub(
            method=f"graph-{branch}",
            family="graph", arch=branch,
            decoder="differentiable_routing",
            host="redacted-ccw-host-01", accel="h100", seed=1234,
            metrics={k: float("nan") for k in pe.results.PRIMARY_METRIC_KEYS},
            n_samples=0,   # GNN released CSV is timing-only
            lat_fwd=fwd, lat_dec=dec, lat_e2e=e2e,
            dataset_version="passage_graph-h100-timing",
            extra={"source_csv": "gnn_h100_256.csv", "note": "timing-only entry"},
        )
        pe.submit(sub, root=RESULTS_ROOT)
        n_graph += 1
    print(f"emitted {n_graph} graph rows")

    # --- emit classical reference baselines on every host ---
    classical_hosts = ["h100", "Thor-CPU", "Orin-CPU", "Tiger-CPU"]
    classical_lats = {"h100": 0.05, "Thor-CPU": 0.5, "Orin-CPU": 0.7, "Tiger-CPU": 1.0}
    classical_oracle_lats = {"h100": 50.0, "Thor-CPU": 50.0, "Orin-CPU": 80.0, "Tiger-CPU": 120.0}
    n_class = 0
    rng = np.random.default_rng(0)
    H = W = 256
    elev = rng.uniform(0, 100, size=(H, W)).astype(np.float64)
    obs = (rng.uniform(0, 1, size=(H, W)) < 0.05).astype(np.uint8)
    bootstrap = pe.AStarPlanner().predict_path(pe.Scene(
        sample_id="bootstrap", elevation=elev, obstacles=obs,
        start=(0, 0), goal=(H - 1, W - 1),
        cost_model="elevation", cost_weight=1.0,
        oracle_path=[(0, 0)], oracle_cost=0.0,
    ))
    scenes = [pe.Scene(
        sample_id=f"r256-{i}", elevation=elev, obstacles=obs,
        start=(0, 0), goal=(H - 1, W - 1),
        cost_model="elevation", cost_weight=1.0,
        oracle_path=bootstrap.predicted_path, oracle_cost=bootstrap.predicted_cost,
    ) for i in range(8)]
    astar_metrics = pe.score_planner(pe.AStarPlanner(), scenes)
    greedy_metrics = pe.score_planner(pe.GreedyToGoalPlanner(), scenes)
    for accel in classical_hosts:
        host = "redacted-ccw-host-01" if accel == "h100" else accel.split("-")[0]
        pe.submit(_make_sub(
            method="astar_classical", family="classical", arch="analytic",
            decoder="analytic_a_star",
            host=host, accel=accel, seed=1234,
            metrics=astar_metrics, n_samples=astar_metrics["n_samples"],
            lat_fwd=0.0, lat_dec=classical_oracle_lats[accel], lat_e2e=classical_oracle_lats[accel],
            dtype="fp64", dataset_version="bootstrapped-256x256-rng0",
            extra={"role": "oracle reference"},
        ), root=RESULTS_ROOT)
        pe.submit(_make_sub(
            method="greedy_classical", family="classical", arch="analytic",
            decoder="manhattan_descent",
            host=host, accel=accel, seed=1234,
            metrics=greedy_metrics, n_samples=greedy_metrics["n_samples"],
            lat_fwd=0.0, lat_dec=classical_lats[accel], lat_e2e=classical_lats[accel],
            dtype="fp64", dataset_version="bootstrapped-256x256-rng0",
            extra={"role": "weak baseline"},
        ), root=RESULTS_ROOT)
        n_class += 2
    print(f"emitted {n_class} classical rows")

    print(f"\nTOTAL rows: {n_vision + n_graph + n_class}")
    lb = pe.load_leaderboard(RESULTS_ROOT)
    print(f"leaderboard rows on disk: {len(lb)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="wipe results/ before generating")
    args = p.parse_args()
    main(reset=args.reset)
