# MIT License
#
# Copyright (c) 2026 THALES

"""Submission storage + leaderboard utilities for `passage-eval`.

A *submission* is one row in the leaderboard: one (method, resolution,
cost_model, seed) tuple together with its primary-metric bundle, the
optional per-sample arrays, the four trustworthiness axes (when
computable), and the reproducibility sidecar. The on-disk layout is
intentionally git-friendly:

    <root>/
    ├── leaderboard.csv
    └── runs/
        └── <run_id>/
            ├── metrics.json
            ├── sidecar.json
            ├── trustworthiness.json (optional)
            └── per_sample.npz       (optional, enables paired-bootstrap recompute)

`run_id` is `<ISO-timestamp>--<method>__<resolution>__<cost>__seed<N>`
which sorts chronologically and is unique per submission.

The leaderboard CSV is an *append-only index*: each row carries the
canonical metadata + the six primary-metric values + a relative
pointer to its run directory. If the CSV is lost it can be rebuilt
from the `runs/` tree via :func:`rebuild_leaderboard`.

Visualisation helpers (matplotlib only, no seaborn/plotly) live next
to the storage layer because notebooks consume them as a single
import-and-call pair.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import getpass
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from passage_eval.protocol.discipline import DEFAULT_SEEDS

__all__ = [
    "Submission",
    "submit",
    "load_leaderboard",
    "rebuild_leaderboard",
    "Leaderboard",
    # base plots
    "plot_primary_bars",
    "plot_pareto_accuracy_latency",
    "plot_trustworthiness_radar",
    "plot_pairwise_heatmap",
    # shiny grouped / matrix plots
    "plot_grouped_bars_by_accelerator",
    "plot_latency_heatmap",
    "plot_metric_vs_accelerator",
    "summarise_leaderboard",
    "scoreboard_markdown",
]


# ---------------------------------------------------------------------------
# Submission record
# ---------------------------------------------------------------------------

PRIMARY_METRIC_KEYS: tuple[str, ...] = (
    "cost_regret_median",
    "cost_regret_p90",
    "path_assd",
    "path_hausdorff_p95",
    "success_rate",
    "obstacle_violation_rate",
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in text).strip("-")


@dataclass
class Submission:
    """One leaderboard entry.

    The class collects every piece of metadata a reviewer might want
    to filter or sort by. Defaults pull `host`, `user`, and `python`
    from the current environment; everything else is required.
    """

    # --- identity -----------------------------------------------------------
    method_name: str                       #: short identifier, e.g. "reg-repvit_m1_bellman_vbridge"
    family: str                            #: "dense_regression" | "dense_segmentation" | "graph" | "classical" | "rl" | ...
    architecture: str                      #: backbone / class label, e.g. "repvit_m1"
    decoder_variant: str = "default"       #: e.g. "V_bridge", "BCE+threshold", "edge_mlp_astar"

    # --- benchmark cell -----------------------------------------------------
    resolution: int = 256                  #: square resolution in pixels
    cost_model: str = "elevation"          #: PASSAGE cost model
    cost_weight: float = 1.0
    split: str = "val"                     #: "val" | "test" | "calibration"
    seed: int = DEFAULT_SEEDS[0]
    n_samples: int = 0

    # --- primary metric bundle ---------------------------------------------
    metrics: dict[str, float] = field(default_factory=dict)

    # --- deployment footprint ----------------------------------------------
    host: str = field(default_factory=socket.gethostname)
    user: str = field(default_factory=lambda: os.environ.get("USER", getpass.getuser()))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    edge_accelerator: str | None = None    #: None | "Thor-GPU" | "Orin-GPU" | "Orin-NPU" | "Tiger-iGPU" | "H100"
    batch_size: int = 1
    dtype: str = "fp32"
    latency_fwd_ms_p50: float | None = None
    latency_decoder_ms_p50: float | None = None
    latency_e2e_ms_p50: float | None = None

    # --- versioning + provenance -------------------------------------------
    package_version: str | None = None     #: passage-eval version used to score
    dataset_version: str | None = None     #: e.g. "Anonym-2045/passage:v0.4"
    commit_sha: str | None = None          #: source-tree SHA at scoring time
    extra: dict[str, Any] = field(default_factory=dict)

    # --- generated at submit time ------------------------------------------
    timestamp_utc: str = field(default_factory=_utc_now)

    # --- optional artefacts (paths populated after submit) -----------------
    per_sample_path: str | None = None
    sidecar_path: str | None = None
    trustworthiness_path: str | None = None

    # ----------------------------------------------------------------------
    def host_accelerator_slug(self) -> str:
        """Stable identifier for the (host, accelerator) pair.

        Examples: ``h100``, ``thor-gpu``, ``orin-npu``, ``tiger-igpu``,
        ``thor-cpu``. ``cpu`` if no accelerator was used.
        """
        host = _slug(self.host).lower()
        if self.edge_accelerator:
            return f"{_slug(self.edge_accelerator).lower()}"
        return f"{host}-cpu"

    def run_id(self) -> str:
        """Globally unique run identifier (used as on-disk directory leaf)."""
        return (
            f"{_slug(self.method_name)}__"
            f"seed{self.seed}__"
            f"{self.timestamp_utc}"
        )

    def run_subpath(self) -> Path:
        """Hierarchical sub-path under ``<root>/runs/`` carrying the
        resolution / cost-model / accelerator axes in directory names."""
        return Path(
            f"r{self.resolution}",
            _slug(self.cost_model),
            self.host_accelerator_slug(),
            self.run_id(),
        )

    def to_row(self) -> dict[str, Any]:
        """Flat dict suitable for a CSV row."""
        row = asdict(self)
        for key in PRIMARY_METRIC_KEYS:
            row[key] = float(self.metrics.get(key, float("nan")))
        # Drop nested fields from the flat row.
        row.pop("metrics")
        row["extra_json"] = json.dumps(self.extra, sort_keys=True) if self.extra else ""
        row.pop("extra")
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Submission":
        primary = {k: float(row[k]) for k in PRIMARY_METRIC_KEYS if k in row}
        kwargs = {f.name: row.get(f.name) for f in dataclasses.fields(cls) if f.name not in {"metrics", "extra"}}
        # Coerce ints / floats / strings appropriately.
        for f in dataclasses.fields(cls):
            v = kwargs.get(f.name)
            if v in (None, ""):
                if f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    kwargs[f.name] = f.default_factory()  # type: ignore[misc]
                else:
                    kwargs[f.name] = None
                continue
            if f.type is int and not isinstance(v, int):
                try: kwargs[f.name] = int(float(v))
                except Exception: pass
            if f.type is float and not isinstance(v, float):
                try: kwargs[f.name] = float(v)
                except Exception: pass
        sub = cls(metrics=primary, **{k: v for k, v in kwargs.items() if k != "metrics"})
        if row.get("extra_json"):
            try: sub.extra = json.loads(row["extra_json"])
            except json.JSONDecodeError: pass
        return sub


# ---------------------------------------------------------------------------
# Submit + index helpers
# ---------------------------------------------------------------------------

LEADERBOARD_CSV = "leaderboard.csv"
RUNS_DIR = "runs"


def _atomic_append_csv(csv_path: Path, row: dict[str, Any], fieldnames: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    raise TypeError(f"non-serializable object: {type(obj).__name__}")


def submit(
    submission: Submission,
    *,
    root: str | Path = "results",
    per_sample: Mapping[str, np.ndarray] | None = None,
    sidecar: Any | None = None,
    trustworthiness: Iterable[Any] | None = None,
) -> Path:
    """Append `submission` to the leaderboard and write its run dir.

    Parameters
    ----------
    submission
        The :class:`Submission` record. Its ``run_id()`` determines the
        on-disk subdirectory.
    root
        Leaderboard root (default ``"results"`` relative to cwd).
    per_sample
        Optional per-sample arrays (one numpy array per metric column,
        e.g. ``{"regret": ..., "assd": ..., ...}``). Saved as a single
        compressed NPZ; required if downstream paired-bootstrap is
        wanted.
    sidecar
        Optional :class:`passage_eval.ReproducibilitySidecar` (any
        object exposing ``to_dict()``).
    trustworthiness
        Optional iterable of :class:`TrustworthinessAxis`-like records
        (any object exposing ``as_dict()``).

    Returns
    -------
    Path
        The run directory.
    """
    root = Path(root)
    run_dir = root / RUNS_DIR / submission.run_subpath()
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, {
        "method_name": submission.method_name,
        "family": submission.family,
        "resolution": submission.resolution,
        "cost_model": submission.cost_model,
        "seed": submission.seed,
        "metrics": submission.metrics,
    })

    if per_sample is not None:
        npz_path = run_dir / "per_sample.npz"
        np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in per_sample.items()})
        submission.per_sample_path = str(npz_path.relative_to(root))

    if sidecar is not None:
        sc_path = run_dir / "sidecar.json"
        payload = sidecar.to_dict() if hasattr(sidecar, "to_dict") else dict(sidecar)
        _write_json(sc_path, payload)
        submission.sidecar_path = str(sc_path.relative_to(root))

    if trustworthiness is not None:
        tr_path = run_dir / "trustworthiness.json"
        payload = [
            (axis.as_dict() if hasattr(axis, "as_dict") else dict(axis))
            for axis in trustworthiness
        ]
        _write_json(tr_path, payload)
        submission.trustworthiness_path = str(tr_path.relative_to(root))

    csv_path = root / LEADERBOARD_CSV
    row = submission.to_row()
    fieldnames = list(row.keys())
    _atomic_append_csv(csv_path, row, fieldnames)
    return run_dir


# ---------------------------------------------------------------------------
# Leaderboard reading / rebuilding
# ---------------------------------------------------------------------------

@dataclass
class Leaderboard:
    """Lightweight wrapper around a leaderboard DataFrame."""

    root: Path
    df: "pandas.DataFrame"   # noqa: F821  pandas typed lazily

    def __len__(self) -> int:
        return int(self.df.shape[0])

    def filter(self, **kwargs: Any) -> "Leaderboard":
        df = self.df
        for k, v in kwargs.items():
            if k not in df.columns:
                raise KeyError(f"column not in leaderboard: {k!r}")
            df = df[df[k] == v]
        return Leaderboard(root=self.root, df=df.reset_index(drop=True))

    def best_per_method(
        self,
        *,
        metric: str = "cost_regret_median",
        higher_is_better: bool = False,
    ) -> "pandas.DataFrame":
        df = self.df
        if higher_is_better:
            idx = df.groupby("method_name")[metric].idxmax()
        else:
            idx = df.groupby("method_name")[metric].idxmin()
        return df.loc[idx].reset_index(drop=True)


def load_leaderboard(root: str | Path = "results") -> Leaderboard:
    """Read `<root>/leaderboard.csv` into a :class:`Leaderboard`."""
    import pandas as pd

    root = Path(root)
    csv_path = root / LEADERBOARD_CSV
    if not csv_path.exists():
        return Leaderboard(root=root, df=pd.DataFrame(columns=list(_csv_fieldnames())))
    df = pd.read_csv(csv_path)
    return Leaderboard(root=root, df=df)


def rebuild_leaderboard(root: str | Path = "results") -> Path:
    """Rebuild `<root>/leaderboard.csv` from the `runs/` tree."""
    import pandas as pd

    root = Path(root)
    runs_root = root / RUNS_DIR
    rows: list[dict[str, Any]] = []
    if runs_root.exists():
        for metrics_path in sorted(runs_root.rglob("metrics.json")):
            run_dir = metrics_path.parent
            payload = json.loads(metrics_path.read_text())
            row = {
                "method_name": payload.get("method_name"),
                "family": payload.get("family", ""),
                "resolution": payload.get("resolution"),
                "cost_model": payload.get("cost_model"),
                "seed": payload.get("seed"),
            }
            row.update({k: payload.get("metrics", {}).get(k, float("nan")) for k in PRIMARY_METRIC_KEYS})
            row["run_id"] = str(run_dir.relative_to(runs_root))
            rows.append(row)
    csv_path = root / LEADERBOARD_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _csv_fieldnames() -> tuple[str, ...]:
    """Canonical column order of the leaderboard CSV."""
    placeholder = Submission(
        method_name="placeholder",
        family="classical",
        architecture="dummy",
        metrics={k: 0.0 for k in PRIMARY_METRIC_KEYS},
    )
    return tuple(placeholder.to_row().keys())


# ---------------------------------------------------------------------------
# Visualisations (matplotlib-only)
# ---------------------------------------------------------------------------

_FAMILY_COLOURS = {
    "dense_regression": "#e8743e",
    "dense_segmentation": "#2378c5",
    "graph": "#2ca02c",
    "rl": "#9467bd",
    "classical": "#7a2222",
}


def _family_colour(family: str) -> str:
    return _FAMILY_COLOURS.get(family, "#444444")


def plot_primary_bars(
    df,
    *,
    metric: str = "cost_regret_median",
    higher_is_better: bool = False,
    ax=None,
    title: str | None = None,
):
    """Bar chart of one primary metric across all leaderboard entries.

    `df` is the DataFrame from :class:`Leaderboard.df`. One bar per
    `(method_name, seed)`; bars are coloured by `family`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.0))
    if df.empty:
        ax.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ascending = not higher_is_better
    df_sorted = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    xs = np.arange(len(df_sorted))
    colours = [_family_colour(f) for f in df_sorted["family"]]
    ax.bar(xs, df_sorted[metric].values, color=colours, edgecolor="black", linewidth=0.4)
    labels = [f"{m}\nseed{s}" for m, s in zip(df_sorted["method_name"], df_sorted["seed"])]
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    arrow = "↑" if higher_is_better else "↓"
    ax.set_title(title or f"{metric} {arrow}  (smaller = better)" if not higher_is_better else f"{metric} {arrow}  (larger = better)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    # Family legend.
    families_seen = list(dict.fromkeys(df_sorted["family"].tolist()))
    handles = [plt.Rectangle((0, 0), 1, 1, color=_family_colour(f), label=f) for f in families_seen]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    return ax


def plot_pareto_accuracy_latency(
    df,
    *,
    accuracy_metric: str = "cost_regret_median",
    latency_column: str = "latency_e2e_ms_p50",
    accuracy_higher_is_better: bool = False,
    ax=None,
    title: str | None = None,
):
    """Scatter of accuracy vs end-to-end latency.

    Pareto frontier is highlighted with a thick line connecting the
    non-dominated points.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    sub = df[[accuracy_metric, latency_column, "method_name", "family", "seed"]].copy()
    sub = sub.dropna(subset=[accuracy_metric, latency_column])
    if sub.empty:
        ax.text(0.5, 0.5, "no rows have both accuracy + latency", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    for _, row in sub.iterrows():
        ax.scatter(row[latency_column], row[accuracy_metric],
                   color=_family_colour(row["family"]), s=70,
                   edgecolor="black", linewidth=0.5,
                   label=f"{row['method_name']} (seed{row['seed']})")
        ax.annotate(row["method_name"],
                    (row[latency_column], row[accuracy_metric]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)

    sign = -1.0 if accuracy_higher_is_better else 1.0
    pareto = sub.copy()
    pareto["score"] = sign * pareto[accuracy_metric]
    pareto = pareto.sort_values(latency_column).reset_index(drop=True)
    front: list[tuple[float, float]] = []
    best = float("inf")
    for _, row in pareto.iterrows():
        if row["score"] < best:
            best = row["score"]
            front.append((row[latency_column], row[accuracy_metric]))
    if front:
        xs, ys = zip(*front)
        ax.plot(xs, ys, "k--", linewidth=1.0, alpha=0.6, label="Pareto frontier")

    ax.set_xlabel(f"{latency_column}  (ms, smaller = better)")
    ax.set_ylabel(f"{accuracy_metric}  ({'larger' if accuracy_higher_is_better else 'smaller'} = better)")
    ax.set_title(title or "Accuracy vs end-to-end latency")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def plot_trustworthiness_radar(
    rows: Sequence[Mapping[str, Any]],
    *,
    axes_order: Sequence[str] = ("Stab", "Geo", "ODDe", "Eff"),
    ax=None,
    title: str | None = None,
):
    """Radar plot of the 4 trustworthiness axes for one or more methods.

    Each entry of `rows` is a dict ``{"method_name": ..., "Stab": v, ...}``
    where the axis values are the non-flag scalars from
    :class:`TrustworthinessAxis`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    if not rows:
        ax.text(0.5, 0.5, "no trustworthiness rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    n = len(axes_order)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).tolist()
    theta += theta[:1]

    for row in rows:
        values = [float(row.get(name, 0.0) or 0.0) for name in axes_order]
        values += values[:1]
        ax.plot(theta, values, linewidth=1.5, label=row.get("method_name", "?"))
        ax.fill(theta, values, alpha=0.10)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(list(axes_order))
    ax.set_title(title or "Trustworthiness radar (raw axis values)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05), fontsize=8, frameon=False)
    return ax


def plot_pairwise_heatmap(
    decisions: Mapping[tuple[str, str], str],
    *,
    methods: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
):
    """Heatmap of pairwise decisions.

    `decisions[(a, b)]` is one of ``"PARETO_DOMINANT_A"``,
    ``"PARETO_DOMINANT_B"``, ``"TRADE_OFF"``,
    ``"INDISTINGUISHABLE"`` (string form of
    :class:`passage_eval.PairwiseDecision`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if not decisions:
        ax.text(0.5, 0.5, "no pairwise decisions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    if methods is None:
        seen: list[str] = []
        for a, b in decisions:
            for x in (a, b):
                if x not in seen: seen.append(x)
        methods = seen

    code = {"PARETO_DOMINANT_A": 1, "PARETO_DOMINANT_B": -1, "TRADE_OFF": 0, "INDISTINGUISHABLE": 0}
    mat = np.zeros((len(methods), len(methods)), dtype=np.int64)
    for (a, b), verdict in decisions.items():
        if a in methods and b in methods:
            mat[methods.index(a), methods.index(b)] = code.get(str(verdict).split(".")[-1], 0)
    cmap = ListedColormap(["#7a2222", "#dddddd", "#2ca02c"])
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8)
    ax.set_title(title or "Pairwise decision (row = A, column = B)")
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.7)
    cbar.set_ticklabels(["B dominates A", "Tie / Trade-off", "A dominates B"])
    return ax


# ---------------------------------------------------------------------------
# Shiny visualisations + scoreboard helpers (added in 0.3.1)
# ---------------------------------------------------------------------------

_FAMILY_ORDER = ("classical", "graph", "dense_segmentation", "dense_regression", "rl")


def _accel_order(values: Iterable[str]) -> list[str]:
    """Stable accelerator ordering: server first, then edge GPU, NPU, iGPU, CPU."""
    priority = {
        "h100": 0,
        "thor-gpu": 1, "orin-gpu": 2, "orin-npu": 3, "tiger-igpu": 4,
        "thor-cpu": 5, "orin-cpu": 6, "tiger-cpu": 7,
        "redacted-ccw-host-02": 8,
    }
    return sorted(set(values), key=lambda v: (priority.get(v, 99), v))


def summarise_leaderboard(df, metric: str = "cost_regret_median") -> "pandas.DataFrame":
    """Three-seed mean/std table grouped by (method, accelerator)."""
    import pandas as pd

    if df.empty: return pd.DataFrame()
    cols = ["method_name", "family", "host_accelerator_slug" if "host_accelerator_slug" in df.columns else None]
    # Recompute the slug column from host + edge_accelerator if missing.
    if "host_accelerator_slug" not in df.columns:
        df = df.copy()
        df["host_accelerator_slug"] = df.apply(
            lambda r: (str(r["edge_accelerator"]).lower()
                       if isinstance(r.get("edge_accelerator"), str) and r["edge_accelerator"]
                       else f"{str(r.get('host', 'unknown')).lower()}-cpu"),
            axis=1,
        )
    grp = df.groupby(["method_name", "family", "host_accelerator_slug"]).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n_seeds=(metric, "count"),
        latency_e2e_ms_p50=("latency_e2e_ms_p50", "median"),
        n_samples=("n_samples", "max"),
    ).reset_index()
    grp = grp.sort_values(["family", "method_name", "host_accelerator_slug"])
    return grp


def plot_grouped_bars_by_accelerator(
    df,
    *,
    metric: str = "cost_regret_median",
    higher_is_better: bool = False,
    ax=None,
    title: str | None = None,
):
    """Method-by-accelerator grouped bar chart, with seed-mean ± std error bars."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11.5, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    summary = summarise_leaderboard(df, metric=metric)
    methods = list(summary.method_name.unique())
    accels = _accel_order(summary.host_accelerator_slug.unique())
    n_methods = len(methods)
    n_accels = len(accels)
    width = 0.8 / max(1, n_accels)

    # Build a per-accelerator colour palette (cool -> warm).
    accel_colour_map = dict(zip(accels, plt.colormaps["viridis"](np.linspace(0.10, 0.85, n_accels))))

    xs = np.arange(n_methods)
    for ai, accel in enumerate(accels):
        rows = summary[summary.host_accelerator_slug == accel]
        means = []; stds = []
        for m in methods:
            row = rows[rows.method_name == m]
            means.append(float(row["mean"].iloc[0]) if not row.empty else float("nan"))
            stds.append(float(row["std"].iloc[0]) if not row.empty and not np.isnan(row["std"].iloc[0]) else 0.0)
        ax.bar(xs + (ai - (n_accels-1)/2) * width, means, width=width,
                yerr=stds, label=accel, color=accel_colour_map[accel],
                edgecolor="black", linewidth=0.4, capsize=2.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    arrow = "↑ larger = better" if higher_is_better else "↓ smaller = better"
    ax.set_title(title or f"{metric}  ({arrow})  · grouped by host / accelerator")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(title="host / accelerator", fontsize=8, ncol=min(4, n_accels), frameon=False, loc="upper left")
    return ax


def plot_latency_heatmap(
    df,
    *,
    latency_column: str = "latency_e2e_ms_p50",
    ax=None,
    title: str | None = None,
    log_norm: bool = True,
):
    """Method × accelerator heatmap of end-to-end latency."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    summary = summarise_leaderboard(df, metric=latency_column)
    methods = sorted(summary.method_name.unique())
    accels = _accel_order(summary.host_accelerator_slug.unique())
    matrix = np.full((len(methods), len(accels)), np.nan, dtype=np.float64)
    for _, row in summary.iterrows():
        i = methods.index(row.method_name)
        j = accels.index(row.host_accelerator_slug)
        matrix[i, j] = float(row.latency_e2e_ms_p50) if np.isfinite(row.latency_e2e_ms_p50) else np.nan

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        ax.text(0.5, 0.5, f"no finite {latency_column}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax
    vmin = max(1e-3, finite.min())
    vmax = finite.max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_norm and vmax / vmin > 5 else Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "viridis", norm=norm, aspect="auto")
    ax.set_xticks(range(len(accels))); ax.set_xticklabels(accels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    ax.set_title(title or f"{latency_column}  · method × accelerator (log-scale colour)")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{latency_column}  (ms)")
    # Annotate cells with rounded values.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7,
                        color="white" if v > finite.mean() else "black")
    ax.set_xlabel("host / accelerator")
    ax.set_ylabel("method")
    return ax


def plot_metric_vs_accelerator(
    df,
    *,
    metric: str = "cost_regret_median",
    ax=None,
    title: str | None = None,
):
    """One line per method, x = accelerator, y = primary metric.

    Useful for spotting accelerator-specific accuracy regressions
    (e.g. the Orin-NPU regression of the regression decoder).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return ax

    summary = summarise_leaderboard(df, metric=metric)
    accels = _accel_order(summary.host_accelerator_slug.unique())
    method_styles = {
        m: dict(marker=mk, color=_family_colour(fam))
        for m, mk, fam in [
            ("seg-convnextv2_atto_baseline", "o", "dense_segmentation"),
            ("reg-repvit_m1_bellman_vbridge", "s", "dense_regression"),
            ("graph-edge_mlp_classical_astar", "D", "graph"),
            ("graph-vmpnn", "^", "graph"),
            ("astar_classical", "P", "classical"),
            ("greedy_classical", "x", "classical"),
        ]
    }

    for method, sub in summary.groupby("method_name"):
        style = method_styles.get(method, {"marker": "o", "color": _family_colour(sub.family.iloc[0])})
        ys = []
        for accel in accels:
            row = sub[sub.host_accelerator_slug == accel]
            ys.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        ax.plot(accels, ys, label=method, marker=style["marker"], color=style["color"], linewidth=1.5)

    ax.set_xticks(range(len(accels)))
    ax.set_xticklabels(accels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric}  vs. host / accelerator")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="best", frameon=False)
    return ax


def scoreboard_markdown(df, *, metric: str = "cost_regret_median",
                        higher_is_better: bool = False) -> str:
    """Render a markdown scoreboard, sorted by `metric` per-family.

    Includes a top header row + per-family sub-tables. Suitable to drop
    into ``results/SCOREBOARD.md`` or a notebook's markdown cell.
    """
    import pandas as pd

    if df.empty:
        return "_(no leaderboard rows)_"

    summary = summarise_leaderboard(df, metric=metric)
    summary = summary.sort_values("mean", ascending=not higher_is_better)
    lines: list[str] = []
    lines.append("# PASSAGE leaderboard scoreboard")
    lines.append("")
    lines.append(f"Metric: **{metric}**, ordered "
                 + ("descending" if higher_is_better else "ascending")
                 + f"; resolutions = {sorted(df.resolution.unique().tolist())}, "
                 + f"cost models = {sorted(df.cost_model.unique().tolist())}.")
    lines.append("")
    families = [f for f in _FAMILY_ORDER if f in summary.family.unique()]
    other = [f for f in summary.family.unique() if f not in _FAMILY_ORDER]
    for family in families + other:
        sub = summary[summary.family == family]
        if sub.empty: continue
        lines.append(f"## family: `{family}`")
        lines.append("")
        lines.append("| method | host / accel | seeds | "
                     f"{metric} (mean ± std) | latency_e2e_ms_p50 | n_samples |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            std = r["std"]
            std_str = f"± {std:.4f}" if pd.notna(std) and std > 0 else ""
            lines.append(
                f"| `{r.method_name}` | `{r.host_accelerator_slug}` | "
                f"{int(r.n_seeds)} | {r['mean']:.4f} {std_str} | "
                f"{r['latency_e2e_ms_p50']:.2f} | {int(r['n_samples'])} |"
            )
        lines.append("")
    return "\n".join(lines)

