# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

`run_id` contains the method, seed and UTC timestamp. A repeated identifier
at the same run path is rejected, so no submission can overwrite another.

The leaderboard CSV is an *append-only index*: each row carries the
canonical metadata + the six primary-metric values + a relative
pointer to its run directory. The CSV can be rebuilt at any time
from the `runs/` tree via :func:`rebuild_leaderboard`.

Visualisation helpers (matplotlib only, no seaborn/plotly) live next
to the storage layer because notebooks consume them as a single
import-and-call pair.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import fcntl
import getpass
import json
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TYPE_CHECKING, get_args, get_type_hints

if TYPE_CHECKING:
    from pandas import DataFrame

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
    method_name: str  #: short identifier, e.g. "reg-repvit_m1_bellman_vbridge"
    family: str  #: "dense_regression" | "dense_segmentation" | "graph" | "classical" | "rl" | ...
    architecture: str  #: backbone / class label, e.g. "repvit_m1"
    decoder_variant: str = "default"  #: e.g. "V_bridge", "BCE+threshold", "edge_mlp_astar"

    # --- benchmark cell -----------------------------------------------------
    resolution: int = 256  #: square resolution in pixels
    cost_model: str = "elevation"  #: PASSAGE cost model
    cost_weight: float = 1.0
    split: str = "val"  #: "val" | "test" | "calibration"
    seed: int = DEFAULT_SEEDS[0]
    n_samples: int = 0

    # --- primary metric bundle ---------------------------------------------
    metrics: dict[str, float] = field(default_factory=dict)

    # --- deployment footprint ----------------------------------------------
    host: str = field(default_factory=socket.gethostname)
    user: str = field(default_factory=lambda: os.environ.get("USER", getpass.getuser()))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    edge_accelerator: str | None = None  #: None | "Thor-GPU" | "Orin-GPU" | "Orin-NPU" | "Tiger-iGPU" | "H100"
    batch_size: int = 1
    dtype: str = "fp32"
    latency_fwd_ms_p50: float | None = None
    latency_decoder_ms_p50: float | None = None
    latency_e2e_ms_p50: float | None = None

    # --- versioning + provenance -------------------------------------------
    package_version: str | None = None  #: passage-eval version used to score
    dataset_version: str | None = None  #: e.g. "thalesgroup/passage:v1.0"
    commit_sha: str | None = None  #: source-tree SHA at scoring time
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
        """Run directory leaf; :func:`submit` refuses an existing path."""
        return f"{_slug(self.method_name)}__seed{self.seed}__{self.timestamp_utc}"

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
        """Restore CSV types, rejecting malformed numeric metadata."""
        primary = {k: float(row[k]) for k in PRIMARY_METRIC_KEYS if row.get(k) not in (None, "")}
        kwargs = {}
        hints = get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if f.name in {"metrics", "extra"}:
                continue
            v = row.get(f.name)
            if v in (None, "") or isinstance(v, (float, np.floating)) and np.isnan(v):
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    kwargs[f.name] = None
                continue
            hint = hints[f.name]
            if hint is int:
                try:
                    number = Decimal(str(v))
                    if not number.is_finite() or number != number.to_integral_value():
                        raise ValueError("expected an integer")
                    v = int(number)
                except (InvalidOperation, ValueError) as exc:
                    raise ValueError(f"Invalid integer field {f.name}: {v!r}") from exc
            elif hint is float or float in get_args(hint):
                try:
                    v = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid numeric field {f.name}: {v!r}") from exc
            kwargs[f.name] = v
        sub = cls(metrics=primary, **kwargs)
        extra_json = row.get("extra_json")
        if extra_json and not (isinstance(extra_json, (float, np.floating)) and np.isnan(extra_json)):
            sub.extra = json.loads(extra_json)
            if not isinstance(sub.extra, dict):
                raise ValueError("extra_json must encode an object")
        return sub


# ---------------------------------------------------------------------------
# Submit + index helpers
# ---------------------------------------------------------------------------

LEADERBOARD_CSV = "leaderboard.csv"
RUNS_DIR = "runs"


@contextmanager
def _store_lock(root: Path):
    """Serialize writers on the supported POSIX filesystem."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".leaderboard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _replace_csv(csv_path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Publish a complete CSV, leaving the previous index intact on failure."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".leaderboard-", suffix=".csv", dir=csv_path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, csv_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_append_csv(csv_path: Path, row: dict[str, Any], fieldnames: Sequence[str]) -> None:
    # Callers hold _store_lock across validation, run publication and index update.
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(dict.fromkeys([*(reader.fieldnames or []), *fieldnames]))
            rows.extend(reader)
    rows.append(row)
    _replace_csv(csv_path, rows, fieldnames)


def _index_contains_row(csv_path: Path, row: Mapping[str, Any]) -> bool:
    """Resolve an interrupted replace from the committed index under its lock."""
    if not csv_path.exists():
        return False
    expected = {key: "" if value is None else str(value) for key, value in row.items()}
    with csv_path.open(newline="", encoding="utf-8") as store:
        return any(
            all(candidate.get(key) == value for key, value in expected.items()) for candidate in csv.DictReader(store)
        )


def _run_metrics_paths(root: Path) -> Iterable[Path]:
    """Visit each run bundle once, excluding its nested execution artifacts."""
    for directory, children, files in os.walk(root / RUNS_DIR):
        children.sort()
        if "metrics.json" in files:
            yield Path(directory) / "metrics.json"
            children.clear()


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

    Notes
    -----
    Writers and rebuilds share a POSIX lock. Existing run paths are rejected.
    A write failure before index publication leaves the previous index unchanged.
    An interruption after publication preserves the committed run and index.
    After an abrupt process interruption between run and index publication,
    :func:`rebuild_leaderboard` recovers the completed run from ``metrics.json``.
    """
    root = Path(root)
    csv_path = root / LEADERBOARD_CSV
    # Build generated pointers on a copy: failure must not mutate caller metadata.
    stored = dataclasses.replace(submission, per_sample_path=None, sidecar_path=None, trustworthiness_path=None)
    contract = stored.extra.get("metric_contract")
    if per_sample is not None and "metric_contract" in per_sample:
        array_contract = np.asarray(per_sample["metric_contract"])
        if array_contract.ndim != 0 or not isinstance(array_contract.item(), str):
            raise ValueError("per_sample metric_contract must be a scalar string")
        if array_contract.item() != contract:
            raise ValueError("per_sample metric contract differs from submission metadata")
    run_dir = root / RUNS_DIR / stored.run_subpath()
    with _store_lock(root):
        if run_dir.exists():
            raise FileExistsError(f"Submission run already exists: {run_dir}")
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as store:
                for row in csv.DictReader(store):
                    existing = json.loads(row.get("extra_json") or "{}")
                    if existing.get("metric_contract") != contract:
                        raise ValueError("Cannot mix metric contracts in one leaderboard; use a fresh results root")
        # The CSV is only an index: a missing/stale index cannot hide runs from
        # contract validation (including a run committed before interruption).
        for metrics_path in _run_metrics_paths(root):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "submission" in payload:
                existing = json.loads(payload["submission"].get("extra_json") or "{}")
            else:
                sidecar_path = metrics_path.parent / "sidecar.json"
                existing = (
                    json.loads(sidecar_path.read_text(encoding="utf-8")).get("extra", {})
                    if sidecar_path.exists()
                    else {}
                )
            if existing.get("metric_contract") != contract:
                raise ValueError("Cannot mix metric contracts in one leaderboard; use a fresh results root")
        # Staging lives outside runs/, so rebuild never indexes incomplete work.
        with tempfile.TemporaryDirectory(prefix=".submission-", dir=root) as temporary:
            staged = Path(temporary) / "run"
            staged.mkdir()
            if per_sample is not None:
                np.savez_compressed(staged / "per_sample.npz", **{k: np.asarray(v) for k, v in per_sample.items()})
                stored.per_sample_path = str((run_dir / "per_sample.npz").relative_to(root))
            if sidecar is not None:
                payload = sidecar.to_dict() if hasattr(sidecar, "to_dict") else dict(sidecar)
                if "metric_contract" in payload.get("extra", {}) and payload["extra"]["metric_contract"] != contract:
                    raise ValueError("sidecar metric contract differs from submission metadata")
                _write_json(staged / "sidecar.json", payload)
                stored.sidecar_path = str((run_dir / "sidecar.json").relative_to(root))
            if trustworthiness is not None:
                payload = [(axis.as_dict() if hasattr(axis, "as_dict") else dict(axis)) for axis in trustworthiness]
                _write_json(staged / "trustworthiness.json", payload)
                stored.trustworthiness_path = str((run_dir / "trustworthiness.json").relative_to(root))
            row = stored.to_row()
            _write_json(
                staged / "metrics.json",
                {
                    "method_name": stored.method_name,
                    "family": stored.family,
                    "resolution": stored.resolution,
                    "cost_model": stored.cost_model,
                    "seed": stored.seed,
                    "metrics": stored.metrics,
                    "submission": row,
                },
            )
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, run_dir)
            try:
                _atomic_append_csv(csv_path, row, list(row))
            except BaseException:
                # A signal can arrive after os.replace committed the index but
                # before the call returned. Never remove a published row's run.
                try:
                    published = _index_contains_row(csv_path, row)
                except (OSError, csv.Error):
                    # Preserve authoritative artifacts if index state is unknown;
                    # an explicit rebuild can recover them after the failure.
                    published = True
                if not published:
                    shutil.rmtree(run_dir)
                raise
        submission.per_sample_path = stored.per_sample_path
        submission.sidecar_path = stored.sidecar_path
        submission.trustworthiness_path = stored.trustworthiness_path
    return run_dir


# ---------------------------------------------------------------------------
# Leaderboard reading / rebuilding
# ---------------------------------------------------------------------------


@dataclass
class Leaderboard:
    """Lightweight wrapper around a leaderboard DataFrame."""

    root: Path
    df: "pandas.DataFrame"  # noqa: F821  pandas typed lazily

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
    ) -> "DataFrame":
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
    """Atomically rebuild the index while excluding concurrent submissions."""
    root = Path(root)
    with _store_lock(root):
        return _rebuild_leaderboard(root)


def _rebuild_leaderboard(root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for metrics_path in _run_metrics_paths(root):
        run_dir = metrics_path.parent
        payload = json.loads(metrics_path.read_text())
        row = {key: None for key in _csv_fieldnames()}
        if "submission" in payload:
            row.update(payload["submission"])
        else:
            # Historical bundles have fewer fields. Preserve unknowns;
            # recover contract provenance from a sidecar when available.
            row.update({k: payload.get(k) for k in row if k in payload})
            sidecar_path = run_dir / "sidecar.json"
            if sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text())
                row["extra_json"] = json.dumps(sidecar.get("extra", {}), sort_keys=True)
                row["sidecar_path"] = str(sidecar_path.relative_to(root))
            row.update({k: payload.get("metrics", {}).get(k, float("nan")) for k in PRIMARY_METRIC_KEYS})
        rows.append({key: row.get(key) for key in _csv_fieldnames()})
    csv_path = root / LEADERBOARD_CSV
    _replace_csv(csv_path, rows, _csv_fieldnames())
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
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    arrow = "↑" if higher_is_better else "↓"
    ax.set_title(
        title or f"{metric} {arrow}  (smaller = better)"
        if not higher_is_better
        else f"{metric} {arrow}  (larger = better)"
    )
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
        ax.set_axis_off()
        return ax

    sub = df[[accuracy_metric, latency_column, "method_name", "family", "seed"]].copy()
    sub = sub.dropna(subset=[accuracy_metric, latency_column])
    if sub.empty:
        ax.text(0.5, 0.5, "no rows have both accuracy + latency", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    for _, row in sub.iterrows():
        ax.scatter(
            row[latency_column],
            row[accuracy_metric],
            color=_family_colour(row["family"]),
            s=70,
            edgecolor="black",
            linewidth=0.5,
            label=f"{row['method_name']} (seed{row['seed']})",
        )
        ax.annotate(
            row["method_name"],
            (row[latency_column], row[accuracy_metric]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

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
        ax.set_axis_off()
        return ax

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
        ax.set_axis_off()
        return ax

    if methods is None:
        seen: list[str] = []
        for a, b in decisions:
            for x in (a, b):
                if x not in seen:
                    seen.append(x)
        methods = seen

    code = {"PARETO_DOMINANT_A": 1, "PARETO_DOMINANT_B": -1, "TRADE_OFF": 0, "INDISTINGUISHABLE": 0}
    mat = np.zeros((len(methods), len(methods)), dtype=np.int64)
    for (a, b), verdict in decisions.items():
        if a in methods and b in methods:
            mat[methods.index(a), methods.index(b)] = code.get(str(verdict).split(".")[-1], 0)
    cmap = ListedColormap(["#7a2222", "#dddddd", "#2ca02c"])
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title(title or "Pairwise decision (row = A, column = B)")
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.7)
    cbar.set_ticklabels(["B dominates A", "Tie / Trade-off", "A dominates B"])
    return ax


# ---------------------------------------------------------------------------
# Shiny visualisations + scoreboard helpers
# ---------------------------------------------------------------------------

_FAMILY_ORDER = ("classical", "graph", "dense_segmentation", "dense_regression", "rl")


def _accel_order(values: Iterable[str]) -> list[str]:
    """Stable accelerator ordering: server first, then edge GPU, NPU, iGPU, CPU."""
    priority = {
        "h100": 0,
        "thor-gpu": 1,
        "orin-gpu": 2,
        "orin-npu": 3,
        "tiger-igpu": 4,
        "thor-cpu": 5,
        "orin-cpu": 6,
        "tiger-cpu": 7,
        "hpc-host-cpu": 8,
    }
    return sorted(set(values), key=lambda v: (priority.get(v, 99), v))


def summarise_leaderboard(
    df,
    metric: str = "cost_regret_median",
    *,
    replicate_policy: Literal["error", "mean"] = "error",
) -> "DataFrame":
    """Mean/std across distinct seeds within identical benchmark conditions.

    Conditions include the method/decoder, data revision and cohort, objective,
    split, runtime, hardware and metric contract. Additional ``extra`` settings
    are retained in ``comparison_extra_json``. Measured timings, per-seed
    checkpoint identities and run pointers are provenance, not conditions;
    every source row is retained in ``provenance_json``. Checkpoint/bundle and
    seed keys in ``inference_kwargs``, and ``component_control.checkpoint_id``,
    are treated as per-seed identities.

    Repeated seeds raise by default. With the explicit ``"mean"`` policy,
    first average replicates within each seed, then weight distinct seeds
    equally. ``n_runs`` counts submitted rows; ``n_seeds`` counts unique seeds;
    ``n_metric_seeds`` counts seeds with a finite estimate. Missing estimates
    remain missing. Timing is the median of per-seed median timings.
    """
    import pandas as pd

    if replicate_policy not in {"error", "mean"}:
        raise ValueError("replicate_policy must be 'error' or 'mean'")
    if df.empty:
        return pd.DataFrame()
    df = df.copy().reset_index(drop=True)
    if "seed" not in df:
        raise ValueError("A seed is required for every leaderboard row")
    seeds = pd.to_numeric(df["seed"], errors="raise")
    if not np.isfinite(seeds).all() or (seeds % 1 != 0).any():
        raise ValueError("Every leaderboard seed must be a finite integer")
    df["seed"] = seeds.astype("int64")
    df[metric] = pd.to_numeric(df[metric], errors="raise").replace([np.inf, -np.inf], np.nan)
    if "latency_e2e_ms_p50" not in df:
        df["latency_e2e_ms_p50"] = np.nan
    # Recompute the slug column from host + edge_accelerator if missing.
    if "host_accelerator_slug" not in df.columns:
        df["host_accelerator_slug"] = df.apply(
            lambda r: (
                _slug(str(r["edge_accelerator"])).lower()
                if isinstance(r.get("edge_accelerator"), str) and r["edge_accelerator"]
                else f"{_slug(str(r.get('host', 'unknown'))).lower()}-cpu"
            ),
            axis=1,
        )
    conditions = [
        "method_name",
        "family",
        "architecture",
        "decoder_variant",
        "resolution",
        "cost_model",
        "cost_weight",
        "split",
        "dataset_version",
        "package_version",
        "commit_sha",
        "host",
        "host_accelerator_slug",
        "batch_size",
        "dtype",
        "n_samples",
        "metric_contract",
        "task_variant",
        "dataset_repo",
        "dataset_revision",
        "comparison_extra_json",
    ]
    extra_columns = {"metric_contract", "task_variant", "dataset_repo", "dataset_revision"}
    # These vary by seed/run, or are measured outcomes rather than settings.
    provenance_keys = {
        "seed",
        "seeds",
        "seed_set",
        "procedure_id",
        "checkpoint_hashes",
        "throughput_samples_per_s",
        "latency_e2e_ms_p50",
        "latency_e2e_ms_p95",
        "latency_e2e_ms_p99",
        "latency_e2e_censored",
        "validation_recost_ms",
        "latency_fwd_ms_p50",
        "latency_decoder_ms_p50",
    }
    for name in conditions:
        if name not in df:
            df[name] = None
    provenance = []
    for index, row in df.iterrows():
        value = row.get("extra_json")
        extra = json.loads(value) if isinstance(value, str) and value else {}
        if not isinstance(extra, dict):
            raise ValueError("extra_json must encode an object")
        comparison = {key: value for key, value in extra.items() if key not in provenance_keys | extra_columns}
        if isinstance(comparison.get("inference_kwargs"), dict):
            comparison["inference_kwargs"] = {
                key: value
                for key, value in comparison["inference_kwargs"].items()
                if "checkpoint" not in key
                and "bundle" not in key
                and key not in {"seed", "random_seed", "training_seed"}
            }
        if isinstance(comparison.get("component_control"), dict):
            comparison["component_control"] = {
                key: value for key, value in comparison["component_control"].items() if key != "checkpoint_id"
            }
        for name in extra_columns:
            df.at[index, name] = extra.get(name, row.get(name))
        df.at[index, "comparison_extra_json"] = json.dumps(comparison, sort_keys=True, default=_json_default)
        provenance.append(
            {
                "seed": int(row["seed"]),
                "timestamp_utc": row.get("timestamp_utc"),
                "per_sample_path": row.get("per_sample_path"),
                "sidecar_path": row.get("sidecar_path"),
                "trustworthiness_path": row.get("trustworthiness_path"),
                "extra": extra,
            }
        )
    df["_provenance"] = provenance
    rows = []
    for keys, group in df.groupby(conditions, dropna=False, sort=False):
        if group.seed.duplicated().any() and replicate_policy == "error":
            raise ValueError(
                "Repeated seed within identical benchmark conditions; "
                "select runs or explicitly set replicate_policy='mean'"
            )
        estimates = group.groupby("seed", sort=False)[metric].mean()
        timings = group.groupby("seed", sort=False).latency_e2e_ms_p50.median()
        rows.append(
            dict(
                zip(conditions, keys),
                mean=float(estimates.mean()),
                std=float(estimates.std()),
                n_seeds=int(group.seed.nunique()),
                n_runs=len(group),
                n_metric_seeds=int(estimates.count()),
                latency_e2e_ms_p50=float(timings.median()) if timings.notna().any() else float("nan"),
                provenance_json=json.dumps(group._provenance.tolist(), sort_keys=True, default=_json_default),
            )
        )
    return pd.DataFrame(rows).sort_values(["family", "method_name", "host_accelerator_slug"], kind="stable")


def _require_unique_plot_conditions(summary: "DataFrame") -> None:
    if summary.duplicated(["method_name", "host_accelerator_slug"]).any():
        raise ValueError(
            "Filter benchmark conditions before plotting: multiple conditions occupy the same method/accelerator cell"
        )
    shared = [
        "resolution",
        "cost_model",
        "cost_weight",
        "split",
        "dataset_version",
        "package_version",
        "n_samples",
        "metric_contract",
        "task_variant",
        "dataset_repo",
        "dataset_revision",
    ]
    if len(summary[shared].drop_duplicates()) > 1:
        raise ValueError(
            "Filter benchmark conditions before plotting: benchmark populations or metric contracts differ"
        )
    stats_and_hardware = {
        "host",
        "host_accelerator_slug",
        "mean",
        "std",
        "n_seeds",
        "n_runs",
        "n_metric_seeds",
        "latency_e2e_ms_p50",
        "provenance_json",
    }
    conditions = summary.drop(columns=list(stats_and_hardware)).copy()
    conditions["comparison_extra_json"] = conditions.comparison_extra_json.map(
        lambda value: json.dumps(
            {key: item for key, item in json.loads(value).items() if key != "hardware_stack"}, sort_keys=True
        )
    )
    for _, group in conditions.groupby("method_name", sort=False):
        if len(group.drop_duplicates()) > 1:
            raise ValueError("Filter benchmark conditions before plotting: method settings differ across accelerators")
    # Cohort identities can be carried in extra rather than Submission fields.
    for key in (
        "sample_ids",
        "dataset_content_hash",
        "reference_identity",
        "evaluator_source_hash",
        "reference_policy",
        "reference_cohort_id",
        "quality_repeat_index",
    ):
        values = {json.dumps(json.loads(value).get(key), sort_keys=True) for value in summary.comparison_extra_json}
        if len(values) > 1:
            raise ValueError(f"Filter benchmark conditions before plotting: {key} differs")


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
        ax.set_axis_off()
        return ax

    summary = summarise_leaderboard(df, metric=metric)
    _require_unique_plot_conditions(summary)
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
        means = []
        stds = []
        for m in methods:
            row = rows[rows.method_name == m]
            means.append(float(row["mean"].iloc[0]) if not row.empty else float("nan"))
            stds.append(float(row["std"].iloc[0]) if not row.empty and not np.isnan(row["std"].iloc[0]) else 0.0)
        ax.bar(
            xs + (ai - (n_accels - 1) / 2) * width,
            means,
            width=width,
            yerr=stds,
            label=accel,
            color=accel_colour_map[accel],
            edgecolor="black",
            linewidth=0.4,
            capsize=2.0,
        )

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
        ax.set_axis_off()
        return ax

    summary = summarise_leaderboard(df, metric=latency_column)
    _require_unique_plot_conditions(summary)
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
        ax.set_axis_off()
        return ax
    vmin = max(1e-3, finite.min())
    vmax = finite.max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_norm and vmax / vmin > 5 else Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "viridis", norm=norm, aspect="auto")
    ax.set_xticks(range(len(accels)))
    ax.set_xticklabels(accels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_title(title or f"{latency_column}  · method × accelerator (log-scale colour)")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{latency_column}  (ms)")
    # Annotate cells with rounded values.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > finite.mean() else "black",
                )
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
        ax.set_axis_off()
        return ax

    summary = summarise_leaderboard(df, metric=metric)
    _require_unique_plot_conditions(summary)
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


def scoreboard_markdown(df, *, metric: str = "cost_regret_median", higher_is_better: bool = False) -> str:
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
    lines.append(
        f"Metric: **{metric}**, ordered "
        + ("descending" if higher_is_better else "ascending")
        + f"; resolutions = {sorted(df.resolution.unique().tolist())}, "
        + f"cost models = {sorted(df.cost_model.unique().tolist())}."
    )
    lines.append("")
    families = [f for f in _FAMILY_ORDER if f in summary.family.unique()]
    other = [f for f in summary.family.unique() if f not in _FAMILY_ORDER]
    for family in families + other:
        sub = summary[summary.family == family]
        if sub.empty:
            continue
        lines.append(f"## family: `{family}`")
        lines.append("")
        lines.append(
            f"| method | host / accel | metric seeds (available / total) | {metric} (mean ± std) | latency_e2e_ms_p50 | n_samples | runs | conditions | sources |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
        for _, r in sub.iterrows():
            std = r["std"]
            std_str = f"± {std:.4f}" if pd.notna(std) and std > 0 else ""
            estimate = f"{r['mean']:.4f} {std_str}".strip() if r.n_metric_seeds else "unavailable"
            conditions = {
                key: (None if pd.isna(value) else value)
                for key, value in r.items()
                if key
                not in {
                    "method_name",
                    "family",
                    "host_accelerator_slug",
                    "mean",
                    "std",
                    "n_seeds",
                    "n_runs",
                    "n_metric_seeds",
                    "latency_e2e_ms_p50",
                    "provenance_json",
                }
            }
            condition_text = json.dumps(conditions, sort_keys=True, default=_json_default).replace("|", "\\|")
            sources = json.dumps(json.loads(r.provenance_json), sort_keys=True).replace("|", "\\|")
            lines.append(
                f"| `{r.method_name}` | `{r.host_accelerator_slug}` | "
                f"{int(r.n_metric_seeds)}/{int(r.n_seeds)} | {estimate} | "
                f"{r['latency_e2e_ms_p50']:.2f} | {int(r['n_samples'])} | {int(r.n_runs)} | "
                f"`{condition_text}` | `{sources}` |"
            )
        lines.append("")
    lines.append(
        "Metric seed coverage is available estimates / total distinct seeds; missing estimates are excluded from the mean and standard deviation."
    )
    return "\n".join(lines)
