"""Render the leaderboard.csv as polished markdown + figures.

Outputs:
    outputs/passage-eval/results/SCOREBOARD.md
    outputs/passage-eval/results/leaderboard.html
    notebooks/passage-eval/figures/leaderboard_grouped_bars.png
    notebooks/passage-eval/figures/leaderboard_latency_heatmap.png
    notebooks/passage-eval/figures/leaderboard_metric_vs_accel.png
    notebooks/passage-eval/figures/leaderboard_pareto.png
    docs/benchmark/leaderboard.md
    docs/assets/leaderboard/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import passage_eval as pe  # noqa: E402

REPO_ROOT = Path("~/projects/passage")
REPO = REPO_ROOT / "passage"
RESULTS_ROOT = REPO / "outputs" / "passage-eval" / "results"
FIGURES_DIR = REPO / "notebooks" / "passage-eval" / "figures"
DOCS_DIR = REPO_ROOT / "passage" / "docs" / "benchmark"
PUBLIC_DOCS_FIG = REPO_ROOT / "passage" / "docs" / "assets" / "leaderboard"
for d in (FIGURES_DIR, DOCS_DIR, PUBLIC_DOCS_FIG):
    d.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "font.family": "DejaVu Sans",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})



_FAMILY_EMOJI_README = {
    "dense_regression": "🧠 regression",
    "dense_segmentation": "🖼️ segmentation",
    "graph": "🕸️ graph",
    "classical": "🧭 classical",
    "rl": "🎓 rl",
}

_FAMILY_BADGE = {
    "dense_regression": '![reg](https://img.shields.io/badge/dense--regression-orange)',
    "dense_segmentation": '![seg](https://img.shields.io/badge/dense--segmentation-blue)',
    "graph": '![graph](https://img.shields.io/badge/graph-2ca02c)',
    "classical": '![classical](https://img.shields.io/badge/classical-7a2222)',
    "rl": '![rl](https://img.shields.io/badge/rl-9467bd)',
}


def _format_pm(val: float, std: float | None) -> str:
    if not np.isfinite(val): return "—"
    s = f"{val:.4f}"
    if std is not None and np.isfinite(std) and std > 0:
        s += f"&nbsp;<sub>±{std:.4f}</sub>"
    return s


def _annotate_summary(df) -> pd.DataFrame:
    summary = pe.summarise_leaderboard(df, metric="cost_regret_median")
    lat = pe.summarise_leaderboard(df, metric="latency_e2e_ms_p50")
    summary = summary.merge(
        lat[["method_name", "host_accelerator_slug", "mean"]].rename(columns={"mean": "latency_mean"}),
        on=["method_name", "host_accelerator_slug"], how="left",
    )
    return summary.sort_values(["family", "mean"]).reset_index(drop=True)


def _build_scoreboard_md(df) -> str:
    summary = _annotate_summary(df)
    n_accels = df["host_accelerator_slug"].nunique() if "host_accelerator_slug" in df.columns else df["edge_accelerator"].nunique()
    lines = [
        f"# PASSAGE leaderboard — `passage-eval` v{pe.__version__}",
        "",
        f"Live snapshot of `results/leaderboard.csv` "
        f"({len(df)} rows · {df['method_name'].nunique()} methods · "
        f"{n_accels} host/accelerator buckets).",
        "",
        "> Resolution = 256², cost-model = `elevation`. Accuracy metrics are "
        "seed-invariant (same checkpoint runs everywhere); the latency column "
        "varies per (host, accelerator).",
        "",
        "## Headline @ 256² / `elevation`",
        "",
        "| family | method | architecture | accelerator | seeds | "
        "CR-med ↓ | latency_e2e_ms_p50 | n |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        sub_df = df[df.method_name == r.method_name]
        m_arch = str(sub_df.architecture.iloc[0]) if not sub_df.empty else ""
        lines.append(
            f"| {_FAMILY_BADGE.get(r.family, r.family)} "
            f"| `{r.method_name}` | `{m_arch}` | `{r.host_accelerator_slug}` "
            f"| {int(r.n_seeds)} | {_format_pm(r['mean'], r['std'])} "
            f"| {r['latency_e2e_ms_p50']:.2f} | {int(r['n_samples'])} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "![grouped bars](../../../notebooks/passage-eval/figures/leaderboard_grouped_bars.png)",
        "",
        "![latency heatmap](../../../notebooks/passage-eval/figures/leaderboard_latency_heatmap.png)",
        "",
        "![metric vs accelerator](../../../notebooks/passage-eval/figures/leaderboard_metric_vs_accel.png)",
        "",
        "![pareto](../../../notebooks/passage-eval/figures/leaderboard_pareto.png)",
        "",
        "---",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run python scripts/passage-eval/generate_256_leaderboard.py --reset",
        "uv run python scripts/passage-eval/render_leaderboard.py",
        "```",
    ])
    return "\n".join(lines)


def _build_docs_leaderboard(df) -> str:
    """Mkdocs page that mirrors the docs/index.md leaderboard preview.

    The interactive plotly tabs + candidate table are sourced from the
    shared partial under ``docs/_partials/leaderboard_main.md``
    produced by ``render_plotly_leaderboard.py``; this page only adds
    the long-form headline-metric table (best per method × accelerator)
    that would be too verbose to fit on the home page.
    """
    summary = _annotate_summary(df)
    family_label = {
        "dense_regression": ":material-image-multiple-outline: regression",
        "dense_segmentation": ":material-image-outline: segmentation",
        "graph": ":material-graph-outline: graph",
        "classical": ":material-cog-outline: classical",
        "rl": ":material-school-outline: rl",
    }
    lines = [
        "---",
        'title: "Leaderboard"',
        "icon: material/trophy-outline",
        "---",
        "",
        "# Leaderboard",
        "",
        "Generated by `scripts/passage-eval/render_leaderboard.py`. "
        "Resolution = `256²`, cost-model = `elevation`.",
        "",
        "## Interactive view",
        "",
        '{% include-markdown "../_partials/leaderboard_main_benchmark.md" %}',
        "",
        "## Full headline table — best per (method × accelerator)",
        "",
        "| Family | Method | Architecture | Accelerator | Seeds | "
        "CR-med ↓ | Latency p50 (ms) | n |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        sub_df = df[df.method_name == r.method_name]
        m_arch = str(sub_df.architecture.iloc[0]) if not sub_df.empty else ""
        std = r["std"]
        std_str = f" ± {std:.4f}" if pd.notna(std) and std > 0 else ""
        lines.append(
            f"| {family_label.get(r.family, r.family)} | `{r.method_name}` "
            f"| `{m_arch}` | `{r.host_accelerator_slug}` | "
            f"{int(r.n_seeds)} | {r['mean']:.4f}{std_str} | "
            f"{r['latency_e2e_ms_p50']:.2f} | {int(r['n_samples'])} |"
        )
    lines.extend([
        "",
        "## How to add an entry",
        "",
        "Use the `passage-eval` CLI — see "
        "[Contributing a method](contributing-a-method.md). Re-run "
        "`make docs-prepare` to refresh the figures + this page.",
    ])
    return "\n".join(lines)



# Minimal HTML dashboard — single file, no external deps.
_HTML_TEMPLATE = """\
<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\">
<title>PASSAGE leaderboard · passage-eval v{VERSION}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
          background: #fafafa; color: #222; margin: 0; padding: 32px; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 6px 0; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 8px 0; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.75rem; font-weight: 600; color: white; }}
  .b-reg {{ background: #e8743e; }}
  .b-seg {{ background: #2378c5; }}
  .b-graph {{ background: #2ca02c; }}
  .b-classical {{ background: #7a2222; }}
  .b-rl {{ background: #9467bd; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-radius: 8px;
           overflow: hidden; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee;
            font-size: 0.88rem; text-align: left; }}
  th {{ background: #f0f0f0; cursor: pointer; }}
  td.right, th.right {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.method, td.accel {{ font-family: ui-monospace, \"SF Mono\", Menlo, monospace; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .grid img {{ width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
  .meta {{ color: #666; font-size: 0.9rem; }}
</style></head>
<body>
  <h1>PASSAGE leaderboard</h1>
  <p class=\"meta\">passage-eval v{VERSION} · {N_ROWS} rows · {N_METHODS} methods · {N_ACCELS} host/accelerator buckets · resolution = 256² · cost = elevation</p>

  <h2>Headline table</h2>
  {TABLE_HTML}

  <h2>Figures</h2>
  <div class=\"grid\">
    <div><strong>Grouped bars · CR-median per (method × accelerator)</strong>
      <img src=\"../../notebooks/passage-eval/figures/leaderboard_grouped_bars.png\"/></div>
    <div><strong>Latency heatmap (log-scale)</strong>
      <img src=\"../../notebooks/passage-eval/figures/leaderboard_latency_heatmap.png\"/></div>
    <div><strong>Metric vs accelerator</strong>
      <img src=\"../../notebooks/passage-eval/figures/leaderboard_metric_vs_accel.png\"/></div>
    <div><strong>Pareto frontier · accuracy vs latency</strong>
      <img src=\"../../notebooks/passage-eval/figures/leaderboard_pareto.png\"/></div>
  </div>
</body></html>
"""


def _build_dashboard_html(df) -> str:
    summary = _annotate_summary(df)
    badge = {
        "dense_regression": '<span class="badge b-reg">regression</span>',
        "dense_segmentation": '<span class="badge b-seg">segmentation</span>',
        "graph": '<span class="badge b-graph">graph</span>',
        "classical": '<span class="badge b-classical">classical</span>',
        "rl": '<span class="badge b-rl">rl</span>',
    }
    rows: list[str] = []
    for _, r in summary.iterrows():
        std = r["std"]
        std_str = f" ± {std:.4f}" if pd.notna(std) and std > 0 else ""
        sub_df = df[df.method_name == r.method_name]
        m_arch = str(sub_df.architecture.iloc[0]) if not sub_df.empty else ""
        rows.append(
            "<tr>"
            f"<td>{badge.get(r.family, r.family)}</td>"
            f"<td class=\"method\">{r.method_name}</td>"
            f"<td class=\"arch\">{m_arch}</td>"
            f"<td class=\"accel\">{r.host_accelerator_slug}</td>"
            f"<td class=\"right\">{int(r.n_seeds)}</td>"
            f"<td class=\"right\">{r['mean']:.4f}{std_str}</td>"
            f"<td class=\"right\">{r['latency_e2e_ms_p50']:.2f}</td>"
            f"<td class=\"right\">{int(r['n_samples'])}</td>"
            "</tr>"
        )
    table_html = (
        "<table><thead><tr>"
        "<th>family</th><th>method</th><th>architecture</th><th>accelerator</th>"
        "<th class=\"right\">seeds</th><th class=\"right\">CR-med ↓</th>"
        "<th class=\"right\">latency_e2e_ms_p50</th><th class=\"right\">n_samples</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    n_accels = df["host_accelerator_slug"].nunique() if "host_accelerator_slug" in df.columns else df["edge_accelerator"].nunique()
    return _HTML_TEMPLATE.format(
        VERSION=pe.__version__, N_ROWS=len(df),
        N_METHODS=df["method_name"].nunique(),
        N_ACCELS=n_accels,
        TABLE_HTML=table_html,
    )


def main() -> None:
    lb = pe.load_leaderboard(RESULTS_ROOT)
    df = lb.df.copy()
    if df.empty:
        print("[render_leaderboard] no rows under", RESULTS_ROOT); return

    df = df[(df.resolution == 256) & (df.cost_model == "elevation")]
    print(f"[render_leaderboard] {len(df)} rows (256² / elevation)")

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    pe.plot_grouped_bars_by_accelerator(df, metric="cost_regret_median", ax=ax)
    fig.tight_layout()
    out = FIGURES_DIR / "leaderboard_grouped_bars.png"
    fig.savefig(out); fig.savefig(PUBLIC_DOCS_FIG / out.name); plt.close(fig)
    print(f"  wrote {out.name}")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    pe.plot_latency_heatmap(df, latency_column="latency_e2e_ms_p50", ax=ax)
    fig.tight_layout()
    out = FIGURES_DIR / "leaderboard_latency_heatmap.png"
    fig.savefig(out); fig.savefig(PUBLIC_DOCS_FIG / out.name); plt.close(fig)
    print(f"  wrote {out.name}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    pe.plot_metric_vs_accelerator(df, metric="cost_regret_median", ax=ax)
    fig.tight_layout()
    out = FIGURES_DIR / "leaderboard_metric_vs_accel.png"
    fig.savefig(out); fig.savefig(PUBLIC_DOCS_FIG / out.name); plt.close(fig)
    print(f"  wrote {out.name}")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    pe.plot_pareto_accuracy_latency(df, accuracy_metric="cost_regret_median",
                                     latency_column="latency_e2e_ms_p50", ax=ax)
    fig.tight_layout()
    out = FIGURES_DIR / "leaderboard_pareto.png"
    fig.savefig(out); fig.savefig(PUBLIC_DOCS_FIG / out.name); plt.close(fig)
    print(f"  wrote {out.name}")

    (RESULTS_ROOT / "SCOREBOARD.md").write_text(_build_scoreboard_md(df))
    print("  wrote results/SCOREBOARD.md")
    (DOCS_DIR / "leaderboard.md").write_text(_build_docs_leaderboard(df))
    print("  wrote docs/benchmark/leaderboard.md")
    (RESULTS_ROOT / "leaderboard.html").write_text(_build_dashboard_html(df))
    print("  wrote results/leaderboard.html")
    _inject_readme(df)




def _inject_readme(df) -> None:
    import re
    df = df.copy()
    df["host_accelerator_slug"] = df.apply(
        lambda r: (str(r["edge_accelerator"]).lower()
                   if isinstance(r.get("edge_accelerator"), str) and str(r["edge_accelerator"]).lower() != "nan" and r["edge_accelerator"]
                   else f"{str(r.get('host', 'unknown')).lower()}-cpu"),
        axis=1,
    )
    summary = pe.summarise_leaderboard(df, metric="cost_regret_median")
    summary = summary.sort_values(["family", "mean"]).reset_index(drop=True)

    rows: list[str] = []
    for _, r in summary.iterrows():
        sub_df = df[df.method_name == r.method_name]
        m_arch = str(sub_df.architecture.iloc[0]) if not sub_df.empty else ""
        std = r["std"]
        std_str = f" ± {std:.4f}" if pd.notna(std) and std > 0 else ""
        rows.append(
            f"<tr><td>{_FAMILY_EMOJI_README.get(r.family, r.family)}</td>"
            f"<td><code>{r.method_name}</code></td>"
            f"<td><code>{m_arch}</code></td>"
            f"<td><code>{r.host_accelerator_slug}</code></td>"
            f"<td align=\"right\">{int(r.n_seeds)}</td>"
            f"<td align=\"right\">{r['mean']:.4f}{std_str}</td>"
            f"<td align=\"right\">{r['latency_e2e_ms_p50']:.2f}</td>"
            f"<td align=\"right\">{int(r['n_samples'])}</td></tr>"
        )

    section_lines = [
        "## 🏆 Leaderboard @ 256² / `elevation`",
        "",
        f"Live snapshot of `outputs/passage-eval/results/leaderboard.csv` "
        f"({len(df)} rows · {df.method_name.nunique()} methods · "
        f"{df.host_accelerator_slug.nunique()} host/accelerator buckets).",
        "Same checkpoint everywhere — only the latency column varies per accelerator.",
        "",
        "<table>",
        "<thead><tr>",
        "<th>family</th><th>method</th><th>architecture</th><th>accelerator</th>",
        "<th align=\"right\">seeds</th><th align=\"right\">CR-med ↓</th>",
        "<th align=\"right\">latency_e2e_ms_p50</th><th align=\"right\">n</th>",
        "</tr></thead>",
        "<tbody>",
        *rows,
        "</tbody>",
        "</table>",
        "",
        "<p align=\"center\">",
        "  <img src=\"notebooks/passage-eval/figures/leaderboard_grouped_bars.png\" alt=\"CR-median per (method × accelerator)\" width=\"880\"/>",
        "</p>",
        "<p align=\"center\">",
        "  <img src=\"notebooks/passage-eval/figures/leaderboard_latency_heatmap.png\" alt=\"Method × accelerator latency heatmap\" width=\"880\"/>",
        "</p>",
        "<p align=\"center\">",
        "  <img src=\"notebooks/passage-eval/figures/leaderboard_pareto.png\" alt=\"Accuracy vs latency Pareto frontier\" width=\"720\"/>",
        "</p>",
        "",
        "Regenerate with",
        "",
        "```bash",
        "uv run python scripts/passage-eval/generate_256_leaderboard.py --reset",
        "uv run python scripts/passage-eval/render_leaderboard.py",
        "```",
        "",
        "More: [`SCOREBOARD.md`](outputs/passage-eval/results/SCOREBOARD.md) · "
        "[`leaderboard.html`](outputs/passage-eval/results/leaderboard.html) · "
        "[Leaderboard documentation](https://thalesgroup.github.io/passage/benchmark/leaderboard/).",
        "",
    ]
    section = "\n".join(section_lines) + "\n\n"

    readme = REPO_ROOT / "passage" / "README.md"
    src = readme.read_text()
    src = re.sub(r"## 🏆 Leaderboard @ 256.*?(?=^## )", "", src, flags=re.DOTALL | re.MULTILINE)
    needle = "## 🗺️ Visual Summary"
    if needle in src:
        src = src.replace(needle, section + needle)
    else:
        src += "\n\n" + section
    readme.write_text(src)
    print("  wrote passage/README.md (leaderboard section refreshed)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# README leaderboard injection (passage repo root)
# ---------------------------------------------------------------------------

_FAMILY_EMOJI_README = {
    "dense_regression": "🧠 regression",
    "dense_segmentation": "🖼️ segmentation",
    "graph": "🕸️ graph",
    "classical": "🧭 classical",
    "rl": "🎓 rl",
}
