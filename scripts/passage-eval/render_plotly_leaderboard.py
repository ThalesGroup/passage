"""Render the per-accelerator Plotly leaderboard panels for the docs site.

Outputs:
    docs/gen/plotly/leaderboard_{accelerator}.html   ← one HTML per accelerator
    docs/gen/plotly/leaderboard_full.html            ← all-in-one grouped bar chart
    docs/_partials/leaderboard_main.md               ← Material tabs partial
    docs/gen/plotly/candidates_table_data.json       ← machine-readable stats

The per-accelerator figures plot CR-median (log-y) vs end-to-end
latency (log-x) for each candidate, with the oracle drawn as a
dashed *vertical reference line* at its latency value (the oracle's
CR is exactly zero by construction and can't be shown on a log axis).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "outputs" / "passage-eval" / "results"
PLOTLY_OUT = REPO_ROOT / "docs" / "gen" / "plotly"
PARTIAL_OUT = REPO_ROOT / "docs" / "_partials"
PLOTLY_OUT.mkdir(parents=True, exist_ok=True)
PARTIAL_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src" / "passage-eval"))
import passage_eval as pe  # noqa: E402


# ---------------------------------------------------------------------------
# Candidate registry — single source of truth for shortnames + descriptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    method_name: str
    shortname: str
    family: str
    description: str
    is_oracle: bool = False
    color: str = "#7a7a7a"


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        method_name="astar_classical",
        shortname="A* (oracle)",
        family="classical",
        description=(
            "Analytic A* — the labeller. The dataset's reference solver. "
            "Scoring it against itself must yield CR=0 by construction; "
            "shown as a dashed vertical line on each plot."
        ),
        is_oracle=True,
        color="#7a2222",
    ),
    Candidate(
        method_name="greedy_classical",
        shortname="greedy",
        family="classical",
        description=(
            "Diagonal-greedy descent that ignores costs. Always feasible, "
            "always suboptimal; smoke baseline."
        ),
        color="#d62728",
    ),
    Candidate(
        method_name="reg-repvit_m1_bellman_vbridge",
        shortname="reg-RepViT",
        family="dense_regression",
        description=(
            "Cost-to-go regression. Backbone: RepViT-M1. Auxiliary loss: "
            "Bellman value-field. Decoder: V_bridge (greedy descent + "
            "Dijkstra fallback)."
        ),
        color="#e8743e",
    ),
    Candidate(
        method_name="seg-convnextv2_atto_baseline",
        shortname="seg-CNXTv2",
        family="dense_segmentation",
        description=(
            "Dense path segmentation. Backbone: ConvNeXtV2-atto. Loss: BCE. "
            "Decoder: threshold + Dijkstra-on-logits."
        ),
        color="#2378c5",
    ),
    Candidate(
        method_name="graph-edge_mlp_classical_astar",
        shortname="gnn-EdgeMLP",
        family="graph",
        description=(
            "Per-edge 5→16→16→1 MLP that predicts edge costs; classical "
            "A* run on the learned cost grid."
        ),
        color="#2ca02c",
    ),
    Candidate(
        method_name="graph-vmpnn",
        shortname="gnn-VMPNN",
        family="graph",
        description=(
            "Vector message-passing neural network (timing-only entry; "
            "predictions not yet wired into the leaderboard)."
        ),
        color="#9467bd",
    ),
)
SHORTNAME = {c.method_name: c.shortname for c in CANDIDATES}
COLOR = {c.method_name: c.color for c in CANDIDATES}
IS_ORACLE = {c.method_name: c.is_oracle for c in CANDIDATES}


ACCELERATOR_ORDER = (
    "h100", "thor-gpu", "orin-gpu", "orin-npu", "tiger-igpu",
    "thor-cpu", "orin-cpu", "tiger-cpu",
)
ACCELERATOR_LABELS = {
    "h100": "H100 (server GPU)",
    "thor-gpu": "Thor (edge GPU)",
    "orin-gpu": "Orin (edge GPU)",
    "orin-npu": "Orin (NPU / DLA)",
    "tiger-igpu": "Tiger-Lake (iGPU)",
    "thor-cpu": "Thor (CPU)",
    "orin-cpu": "Orin (CPU)",
    "tiger-cpu": "Tiger-Lake (CPU)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_df() -> pd.DataFrame:
    lb = pe.load_leaderboard(RESULTS_ROOT)
    df = lb.df.copy()
    df = df[(df["resolution"] == 256) & (df["cost_model"] == "elevation")]
    df["host_accelerator_slug"] = df.apply(
        lambda r: (str(r["edge_accelerator"]).lower()
                   if isinstance(r.get("edge_accelerator"), str) and str(r["edge_accelerator"]).lower() != "nan" and r["edge_accelerator"]
                   else f"{str(r.get('host', 'unknown')).lower()}-cpu"),
        axis=1,
    )
    df["shortname"] = df["method_name"].map(SHORTNAME).fillna(df["method_name"])
    df["color"] = df["method_name"].map(COLOR).fillna("#666666")
    df["is_oracle"] = df["method_name"].map(IS_ORACLE).fillna(False)
    return df


# ---------------------------------------------------------------------------
# Per-accelerator plotly figure (oracle = vertical reference line)
# ---------------------------------------------------------------------------


def _per_accelerator_fig(df_accel: pd.DataFrame, accel: str) -> go.Figure:
    agg = df_accel.groupby("method_name", as_index=False).agg(
        cost_regret_median=("cost_regret_median", "mean"),
        cost_regret_std=("cost_regret_median", "std"),
        latency_e2e_ms_p50=("latency_e2e_ms_p50", "median"),
        shortname=("shortname", "first"),
        color=("color", "first"),
        is_oracle=("is_oracle", "first"),
        n_seeds=("seed", "nunique"),
        n_samples=("n_samples", "max"),
    )

    fig = go.Figure()

    # Candidate traces (oracle excluded — drawn as a vline below).
    cands = agg[~agg["is_oracle"]]
    cr_vals = cands["cost_regret_median"].astype(float).clip(lower=1e-5)
    y_min = float(cr_vals.min()) * 0.5 if len(cr_vals) else 1e-4
    y_max = float(cr_vals.max()) * 2.0 if len(cr_vals) else 1.0

    for _, row in cands.iterrows():
        std = float(row["cost_regret_std"]) if pd.notna(row["cost_regret_std"]) else 0.0
        std_str = f" ± {std:.4f}" if std > 0 else ""
        hover = (
            f"<b>{row['shortname']}</b>"
            f"<br>method = {row['method_name']}"
            f"<br>accel  = {accel}"
            f"<br>CR-med = {float(row['cost_regret_median']):.4f}{std_str}"
            f"<br>latency_e2e_p50 = {float(row['latency_e2e_ms_p50']):.2f} ms"
            f"<br>val samples per seed = {int(row['n_samples'])}"
            f"<br>seeds = {int(row['n_seeds'])}"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[float(row["latency_e2e_ms_p50"])],
                y=[max(float(row["cost_regret_median"]), 1e-5)],
                mode="markers",
                name=row["shortname"],
                marker=dict(
                    size=14, color=row["color"], symbol="circle",
                    line=dict(color="white", width=1.0),
                ),
                hovertemplate=hover,
            )
        )

    # Oracle as a vertical dashed reference line at its latency.
    oracle = agg[agg["is_oracle"]]
    if not oracle.empty:
        x_oracle = float(oracle["latency_e2e_ms_p50"].iloc[0])
        color = str(oracle["color"].iloc[0])
        # Background reference line (no legend entry).
        fig.add_vline(
            x=x_oracle,
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.85,
        )
        # Invisible scatter to add a legend entry for the reference line.
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                name=f"{oracle['shortname'].iloc[0]} latency = {x_oracle:.1f} ms",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{ACCELERATOR_LABELS.get(accel, accel)}</b>",
            x=0.5, xanchor="center",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="latency p50 (ms)",
            type="log",
            showgrid=True, gridcolor="#eee",
            zeroline=False,
        ),
        yaxis=dict(
            title="CR-median ↓",
            type="log",
            range=[
                __import__("math").log10(max(y_min, 1e-6)),
                __import__("math").log10(max(y_max, 1.0)),
            ],
            showgrid=True, gridcolor="#eee",
            zeroline=False,
        ),
        # Horizontal legend on TOP — keeps it well clear of the x-axis title.
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.06,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420,
        margin=dict(l=70, r=15, t=110, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def render_per_accelerator(df: pd.DataFrame) -> list[str]:
    accels = [a for a in ACCELERATOR_ORDER if a in df["host_accelerator_slug"].unique()]
    for accel in accels:
        sub = df[df["host_accelerator_slug"] == accel].copy()
        if sub.empty: continue
        fig = _per_accelerator_fig(sub, accel)
        out = PLOTLY_OUT / f"leaderboard_{accel}.html"
        fig.write_html(
            out, include_plotlyjs="cdn", full_html=True,
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return accels


def render_full(df: pd.DataFrame, out_path: Path) -> None:
    accels = [a for a in ACCELERATOR_ORDER if a in df["host_accelerator_slug"].unique()]
    methods = list(dict.fromkeys(df["method_name"]))

    agg = df.groupby(["method_name", "host_accelerator_slug"], as_index=False).agg(
        cost_regret_median=("cost_regret_median", "mean"),
        latency_e2e_ms_p50=("latency_e2e_ms_p50", "median"),
        shortname=("shortname", "first"),
        color=("color", "first"),
        is_oracle=("is_oracle", "first"),
    )

    fig = go.Figure()
    for method in methods:
        sub = agg[agg["method_name"] == method].copy()
        sub["__order"] = sub["host_accelerator_slug"].map(
            {a: i for i, a in enumerate(accels)}
        )
        sub = sub.sort_values("__order")
        is_oracle = bool(sub["is_oracle"].iloc[0])
        shortname = sub["shortname"].iloc[0]
        color = sub["color"].iloc[0]
        fig.add_trace(
            go.Bar(
                x=sub["host_accelerator_slug"].map(lambda a: ACCELERATOR_LABELS.get(a, a)),
                y=sub["cost_regret_median"].clip(lower=1e-5),
                name=shortname,
                marker=dict(
                    color=color,
                    line=dict(color="black", width=2) if is_oracle else dict(width=0),
                    pattern_shape="/" if is_oracle else None,
                ),
                hovertemplate=(
                    f"<b>{shortname}</b><br>accelerator=%{{x}}<br>"
                    "CR-med=%{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="PASSAGE 256² / elevation — leaderboard by accelerator",
        xaxis_title="host / accelerator",
        yaxis=dict(title="CR-median ↓ (log)", type="log",
                    showgrid=True, gridcolor="#eee"),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=520,
        margin=dict(l=60, r=10, t=70, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11)),
    )

    fig.write_html(
        out_path, include_plotlyjs="cdn", full_html=True,
        config={"displayModeBar": True, "displaylogo": False, "responsive": True},
    )
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Candidate-description + tabbed-figure partial
# ---------------------------------------------------------------------------


def render_partial(df: pd.DataFrame, accels: list[str],
                   out_path: Path, *, iframe_prefix: str) -> dict[str, int]:
    n_candidates = df["method_name"].nunique()
    n_accels = df["host_accelerator_slug"].nunique()
    n_rows = len(df)

    # --- Material tabs, one per accelerator -------------------------------
    tabs: list[str] = []
    for accel in accels:
        label = ACCELERATOR_LABELS.get(accel, accel)
        tabs.append(f'=== "{label}"')
        tabs.append("")
        tabs.append(
            f'    <iframe src="{iframe_prefix}/leaderboard_{accel}.html" '
            'width="100%" height="460" loading="lazy" '
            'style="border:0; background:transparent;" '
            f'title="Leaderboard — {label}"></iframe>'
        )
        tabs.append("")

    # --- Candidate description table --------------------------------------
    rows: list[str] = []
    for c in CANDIDATES:
        if c.method_name not in df["method_name"].values:
            continue
        sub = df[df["method_name"] == c.method_name]
        n_seeds = int(sub["seed"].nunique())
        n_samples = int(sub["n_samples"].max())
        best_cr = float(sub["cost_regret_median"].min())
        if c.is_oracle:
            shortname_html = f"⭐ <strong><em>{c.shortname}</em></strong>"
            row_class = ' class="oracle-row"'
        else:
            shortname_html = f"<code>{c.shortname}</code>"
            row_class = ''
        rows.append(
            f"<tr{row_class}>"
            f"<td>{shortname_html}</td>"
            f"<td><code>{c.method_name}</code></td>"
            f"<td>{c.family.replace('_', ' ')}</td>"
            f"<td>{c.description}</td>"
            f"<td align=\"right\">{n_seeds}</td>"
            f"<td align=\"right\">{n_samples}</td>"
            f"<td align=\"right\">{best_cr:.4f}</td>"
            "</tr>"
        )

    md = f"""<!-- auto-generated by scripts/passage-eval/render_plotly_leaderboard.py — do not edit -->

<div class="stats-bar" markdown>

<div class="stat-item">
  <span class="stat-value">{n_candidates}</span>
  <span class="stat-label">Candidates</span>
</div>
<div class="stat-item">
  <span class="stat-value">{n_accels}</span>
  <span class="stat-label">Host / accelerator buckets</span>
</div>
<div class="stat-item">
  <span class="stat-value">{n_rows}</span>
  <span class="stat-label">Leaderboard rows</span>
</div>

</div>

#### Accuracy vs latency, per accelerator

Each tab shows one host / accelerator. **Coloured circles** mark the
candidates' cost-regret vs end-to-end latency; the **dashed vertical
line** marks the A* oracle's latency (its `CR-median = 0` by
construction). Hover any point to see the full metric breakdown.

{chr(10).join(tabs)}

<p class="leaderboard-fullsize"><a href="{iframe_prefix}/leaderboard_full.html"
   target="_blank" rel="noopener">Open full-size grouped bar chart ↗</a></p>

#### Candidates

<table class="leaderboard-candidates">
<thead><tr>
  <th>Shortname</th>
  <th>Method id</th>
  <th>Family</th>
  <th>Description</th>
  <th align="right" title="Pre-registered training seeds (passage_eval.DEFAULT_SEEDS = 1234, 2026, 3407)">Seeds</th>
  <th align="right" title="Number of validation samples scored per seed">n_samples</th>
  <th align="right">Best CR-med ↓</th>
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>

<p class="leaderboard-legend">
  <strong>n_samples</strong>: number of validation samples scored per
  seed (1024 for the released vision baselines; smaller for the
  classical / smoke baselines). <strong>Best CR-med</strong>:
  smallest seed-mean of <code>cost_regret_median</code> across the
  accelerators this candidate has been benchmarked on.
</p>
"""
    out_path.write_text(md, encoding="utf-8")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  (candidates={n_candidates}, rows={n_rows})")
    return {"n_candidates": n_candidates, "n_accels": n_accels, "n_rows": n_rows}


def main() -> int:
    df = _load_df()
    if df.empty:
        print("[render-plotly] empty leaderboard, skipping")
        return 0
    print(f"[render-plotly] {len(df)} rows / {df['method_name'].nunique()} candidates / "
          f"{df['host_accelerator_slug'].nunique()} accelerators")

    accels = render_per_accelerator(df)
    render_full(df, PLOTLY_OUT / "leaderboard_full.html")
    # Two partials with the right iframe prefix for each consumer page:
    # docs/index.md is at depth 0; docs/benchmark/leaderboard.md renders to
    # site/benchmark/leaderboard/index.html (use_directory_urls=true), so the
    # prefix must climb two levels back up.
    stats = render_partial(
        df, accels,
        PARTIAL_OUT / "leaderboard_main_index.md",
        iframe_prefix="gen/plotly",
    )
    render_partial(
        df, accels,
        PARTIAL_OUT / "leaderboard_main_benchmark.md",
        iframe_prefix="../../gen/plotly",
    )
    (PLOTLY_OUT / "candidates_table_data.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print("[render-plotly] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
