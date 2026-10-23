"""Render the per-accelerator Plotly leaderboard panels for the docs site.

Outputs:
    docs/gen/plotly/leaderboard_{accelerator}.html   ← one HTML per accelerator
    docs/gen/plotly/leaderboard_full.html            ← all-in-one grouped bar chart
    docs/_partials/leaderboard_main.md               ← Material tabs partial
    docs/gen/plotly/candidates_table_data.json       ← machine-readable stats

The released leaderboard (``src/passage-eval/results/leaderboard.csv``) is
read. While it has no entries, the partials announce the entries to come and
no figure is written.

The per-accelerator figures plot CR-median (log-y) vs end-to-end
latency (log-x) for each candidate, with the oracle drawn as a
dashed *vertical reference line* at its latency value (the oracle's
CR is exactly zero by construction and can't be shown on a log axis).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "src" / "passage-eval" / "results"
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
        description=("Diagonal-greedy descent that ignores costs. Always feasible, always suboptimal; smoke baseline."),
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
            "Dense path segmentation. Backbone: ConvNeXtV2-atto. Loss: BCE. Decoder: threshold + Dijkstra-on-logits."
        ),
        color="#2378c5",
    ),
    Candidate(
        method_name="graph-edge_mlp_classical_astar",
        shortname="gnn-EdgeMLP",
        family="graph",
        description=("Per-edge 5→16→16→1 MLP that predicts edge costs; classical A* run on the learned cost grid."),
        color="#2ca02c",
    ),
    Candidate(
        method_name="graph-vmpnn",
        shortname="gnn-VMPNN",
        family="graph",
        description=(
            "Vector message-passing neural network (timing-only entry; predictions not yet wired into the leaderboard)."
        ),
        color="#9467bd",
    ),
)
SHORTNAME = {c.method_name: c.shortname for c in CANDIDATES}
COLOR = {c.method_name: c.color for c in CANDIDATES}
IS_ORACLE = {c.method_name: c.is_oracle for c in CANDIDATES}
REGISTRY = {c.method_name: c for c in CANDIDATES}


ACCELERATOR_ORDER = (
    "h100",
    "thor-gpu",
    "orin-gpu",
    "orin-npu",
    "tiger-igpu",
    "thor-cpu",
    "orin-cpu",
    "tiger-cpu",
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
    if df.empty:
        return df
    df["host_accelerator_slug"] = df.apply(
        lambda r: (
            str(r["edge_accelerator"]).lower()
            if isinstance(r.get("edge_accelerator"), str)
            and str(r["edge_accelerator"]).lower() != "nan"
            and r["edge_accelerator"]
            else f"{str(r.get('host', 'unknown')).lower()}-cpu"
        ),
        axis=1,
    )
    df["shortname"] = df["method_name"].map(SHORTNAME).fillna(df["method_name"])
    df["color"] = df["method_name"].map(COLOR).fillna("#666666")
    df["is_oracle"] = df["method_name"].map(IS_ORACLE).fillna(False)
    return df


def _accelerators(df: pd.DataFrame) -> list[str]:
    present = set(df["host_accelerator_slug"])
    return [a for a in ACCELERATOR_ORDER if a in present] + sorted(present - set(ACCELERATOR_ORDER))


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """Use the storage layer's condition and distinct-seed contract everywhere."""
    summary = pe.summarise_leaderboard(df).copy()
    if summary.empty:
        return summary
    summary["shortname"] = summary["method_name"].map(SHORTNAME).fillna(summary["method_name"])
    summary["color"] = summary["method_name"].map(COLOR).fillna("#666666")
    summary["is_oracle"] = summary["method_name"].map(IS_ORACLE).fillna(False).astype(bool)
    excluded = {
        "method_name",
        "family",
        "host_accelerator_slug",
        "mean",
        "std",
        "n_seeds",
        "n_runs",
        "n_metric_seeds",
        "n_samples",
        "latency_e2e_ms_p50",
        "provenance_json",
        "shortname",
        "color",
        "is_oracle",
    }
    conditions = [column for column in summary.columns if column not in excluded]
    summary["condition"] = summary.apply(
        lambda row: "; ".join(f"{key}={row[key]}" for key in conditions if pd.notna(row[key])), axis=1
    )
    indices = summary.groupby("method_name", sort=False).cumcount() + 1
    summary["label"] = summary["shortname"] + " · condition " + indices.astype(str)
    summary["metric_available"] = summary["mean"].map(math.isfinite) & (summary["n_metric_seeds"] > 0)
    summary["seed_coverage"] = summary["n_metric_seeds"].astype(str) + "/" + summary["n_seeds"].astype(str)
    return summary


def _metric_text(row: pd.Series) -> str:
    if not row["metric_available"]:
        return "Unavailable"
    std = f"{row['std']:.4f}" if pd.notna(row["std"]) else "—"
    return f"{row['mean']:.4f} ± {std}"


def _annotate_unavailable(fig: go.Figure, summary: pd.DataFrame) -> None:
    for index, (_, row) in enumerate(summary[~summary["metric_available"]].iterrows()):
        fig.add_annotation(
            text=f"{escape(row['label'])}: CR-med unavailable (metric seeds={row['seed_coverage']})",
            xref="paper",
            yref="paper",
            x=0,
            y=1 - index * 0.08,
            xanchor="left",
            yanchor="top",
            showarrow=False,
        )


# ---------------------------------------------------------------------------
# Per-accelerator plotly figure (oracle = vertical reference line)
# ---------------------------------------------------------------------------


def _per_accelerator_fig(df_accel: pd.DataFrame, accel: str) -> go.Figure:
    agg = _summary(df_accel)

    fig = go.Figure()
    _annotate_unavailable(fig, agg)

    # Candidate traces (oracle excluded — drawn as a vline below).
    cands = agg[~agg["is_oracle"] & agg["metric_available"]]
    cr_vals = cands["mean"].astype(float).clip(lower=1e-5)
    y_min = float(cr_vals.min()) * 0.5 if len(cr_vals) else 1e-4
    y_max = float(cr_vals.max()) * 2.0 if len(cr_vals) else 1.0

    for _, row in cands.iterrows():
        hover = (
            f"<b>{row['shortname']}</b>"
            f"<br>method = {row['method_name']}"
            f"<br>accel  = {accel}"
            f"<br>CR-med seed mean = {_metric_text(row)}"
            f"<br>latency_e2e_p50 = {float(row['latency_e2e_ms_p50']):.2f} ms"
            f"<br>samples per seed = {int(row['n_samples'])}"
            f"<br>metric seeds={row['seed_coverage']}"
            f"<br>runs = {int(row['n_runs'])}"
            f"<br>{escape(row['condition'])}"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[float(row["latency_e2e_ms_p50"])],
                y=[max(float(row["mean"]), 1e-5)],
                mode="markers",
                name=row["label"],
                marker=dict(
                    size=14,
                    color=row["color"],
                    symbol="circle",
                    line=dict(color="white", width=1.0),
                ),
                hovertemplate=hover,
            )
        )

    # Oracle as a vertical dashed reference line at its latency.
    oracle = agg[agg["is_oracle"] & agg["metric_available"]]
    for _, row in oracle.iterrows():
        x_oracle = float(row["latency_e2e_ms_p50"])
        color = str(row["color"])
        if not math.isfinite(x_oracle):
            continue
        # Background reference line (no legend entry).
        fig.add_vline(
            x=x_oracle,
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.85,
        )
        # Invisible scatter to add a legend entry for the reference line.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{row['label']} latency = {x_oracle:.1f} ms; metric seeds={row['seed_coverage']}",
                line=dict(color=color, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{ACCELERATOR_LABELS.get(accel, accel)}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="latency p50 (ms)",
            type="log",
            showgrid=True,
            gridcolor="#eee",
            zeroline=False,
        ),
        yaxis=dict(
            title="CR-median ↓",
            type="log",
            range=[
                __import__("math").log10(max(y_min, 1e-6)),
                __import__("math").log10(max(y_max, 1.0)),
            ],
            showgrid=True,
            gridcolor="#eee",
            zeroline=False,
        ),
        # Horizontal legend on TOP — keeps it well clear of the x-axis title.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=70, r=15, t=110, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def render_per_accelerator(df: pd.DataFrame) -> list[str]:
    accels = _accelerators(df)
    for accel in accels:
        sub = df[df["host_accelerator_slug"] == accel].copy()
        if sub.empty:
            continue
        fig = _per_accelerator_fig(sub, accel)
        out = PLOTLY_OUT / f"leaderboard_{accel}.html"
        fig.write_html(
            out,
            include_plotlyjs="cdn",
            full_html=True,
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return accels


def render_full(df: pd.DataFrame, out_path: Path) -> None:
    agg = _summary(df)

    fig = go.Figure()
    _annotate_unavailable(fig, agg)
    for _, row in agg[agg["metric_available"]].iterrows():
        is_oracle = bool(row["is_oracle"])
        shortname = row["shortname"]
        color = row["color"]
        accel = row["host_accelerator_slug"]
        fig.add_trace(
            go.Bar(
                x=[ACCELERATOR_LABELS.get(accel, accel)],
                y=[max(float(row["mean"]), 1e-5)],
                name=row["label"],
                marker=dict(
                    color=color,
                    line=dict(color="black", width=2) if is_oracle else dict(width=0),
                    pattern_shape="/" if is_oracle else None,
                ),
                hovertemplate=(
                    f"<b>{shortname}</b><br>accelerator=%{{x}}<br>"
                    f"CR-med seed mean={_metric_text(row)}<br>"
                    f"metric seeds={row['seed_coverage']}; runs={int(row['n_runs'])}<br>"
                    f"{escape(row['condition'])}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="PASSAGE 256² / elevation — leaderboard by accelerator",
        xaxis_title="host / accelerator",
        yaxis=dict(title="CR-median ↓ (log)", type="log", showgrid=True, gridcolor="#eee"),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=60, r=10, t=70, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, font=dict(size=11)),
    )

    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "displaylogo": False, "responsive": True},
    )
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Candidate-description + tabbed-figure partial
# ---------------------------------------------------------------------------


def render_partial(
    df: pd.DataFrame,
    accels: list[str],
    out_path: Path,
    *,
    iframe_prefix: str,
    protocol_href: str = "benchmark/protocol/",
) -> dict[str, Any]:
    n_candidates = df["method_name"].nunique()
    n_accels = df["host_accelerator_slug"].nunique()
    n_rows = len(df)
    summary = _summary(df)

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
    for _, row in summary.iterrows():
        candidate = REGISTRY.get(row["method_name"])
        description = candidate.description if candidate else ""
        if row["is_oracle"]:
            shortname_html = f"⭐ <strong><em>{escape(row['shortname'])}</em></strong>"
            row_class = ' class="oracle-row"'
        else:
            shortname_html = f"<code>{escape(row['shortname'])}</code>"
            row_class = ""
        provenance = escape(str(row.get("provenance_json", "")))
        contract = escape(str(row.get("metric_contract", "unknown")))
        rows.append(
            f"<tr{row_class}>"
            f"<td>{shortname_html}</td>"
            f"<td><code>{escape(row['method_name'])}</code></td>"
            f"<td>{escape(row['family'].replace('_', ' '))}</td>"
            f"<td>{escape(description)}</td>"
            f"<td>{escape(row['host_accelerator_slug'])}<br>{escape(row['condition'])}</td>"
            f'<td><a href="{escape(protocol_href, quote=True)}"><code>{contract}</code></a>'
            f"<details><summary>Run provenance</summary><pre>{provenance}</pre></details></td>"
            f'<td align="right">{row["seed_coverage"]}</td>'
            f'<td align="right">{int(row["n_runs"])}</td>'
            f'<td align="right">{int(row["n_samples"])}</td>'
            f'<td align="right">{_metric_text(row)}</td>'
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
Each point and table row keeps its benchmark conditions, data revision,
code version and metric contract separate. Values are means over distinct
training seeds with available metric values; repeated seeds in the same condition
are rejected. Metric seed coverage is shown as available/total; candidates with
no available metric values are labelled unavailable and have no plotted estimate.

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
  <th>Accelerator / benchmark conditions</th>
  <th>Metric contract / provenance</th>
  <th align="right" title="Seeds with an available metric value / all distinct training seeds">Metric seeds / total</th>
  <th align="right">Runs</th>
  <th align="right" title="Number of samples scored per seed">n_samples</th>
  <th align="right">CR-med seed mean ± std ↓</th>
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>

<p class="leaderboard-legend">
  <strong>n_samples</strong>: number of samples scored per seed in the
  stated condition. <strong>CR-med seed mean ± std</strong>: arithmetic
  mean and sample standard deviation of each available distinct seed's
  <code>cost_regret_median</code>, using the same summary as the graphs.
  <strong>Metric seeds / total</strong> reports available/total distinct seeds;
  0/total is unavailable and has no plotted estimate.
  A single seed has no estimated standard deviation (shown as —).
  See the <a href="{escape(protocol_href, quote=True)}">metric contract</a>
  before comparing rows from different conditions.
</p>
"""
    out_path.write_text(md, encoding="utf-8")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  (candidates={n_candidates}, rows={n_rows})")
    return {
        "n_candidates": n_candidates,
        "n_accels": n_accels,
        "n_rows": n_rows,
        "conditions": json.loads(summary.to_json(orient="records")),
    }


EMPTY_PARTIAL = """<!-- auto-generated by scripts/passage-eval/render_plotly_leaderboard.py — do not edit -->

!!! info "Entries to come"

    The leaderboard has no entries yet. Its first entries will be the paper's
    two dense-vision finalists, retrained on PASSAGE v1.0 and scored once on
    the test split with `passage-eval`; see
    [First entries](../benchmark/candidates.md).
"""


def render_empty() -> None:
    """Announce the entries to come and drop figures of earlier renders."""
    for name in ("leaderboard_main_index.md", "leaderboard_main_benchmark.md"):
        (PARTIAL_OUT / name).write_text(EMPTY_PARTIAL, encoding="utf-8")
    for path in sorted(PLOTLY_OUT.glob("leaderboard_*.html")):
        path.unlink()
    (PLOTLY_OUT / "candidates_table_data.json").unlink(missing_ok=True)


def main() -> int:
    df = _load_df()
    if df.empty:
        render_empty()
        print("[render-plotly] empty leaderboard: wrote the entries-to-come partials")
        return 0
    print(
        f"[render-plotly] {len(df)} rows / {df['method_name'].nunique()} candidates / "
        f"{df['host_accelerator_slug'].nunique()} accelerators"
    )

    accels = render_per_accelerator(df)
    render_full(df, PLOTLY_OUT / "leaderboard_full.html")
    # Two partials with the right iframe prefix for each consumer page:
    # docs/index.md is at depth 0; docs/benchmark/leaderboard.md renders to
    # site/benchmark/leaderboard/index.html (use_directory_urls=true), so the
    # prefix must climb two levels back up.
    stats = render_partial(
        df,
        accels,
        PARTIAL_OUT / "leaderboard_main_index.md",
        iframe_prefix="gen/plotly",
    )
    render_partial(
        df,
        accels,
        PARTIAL_OUT / "leaderboard_main_benchmark.md",
        iframe_prefix="../../gen/plotly",
        protocol_href="../../benchmark/protocol/",
    )
    (PLOTLY_OUT / "candidates_table_data.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[render-plotly] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
