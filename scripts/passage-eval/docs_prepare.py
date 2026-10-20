"""Execute documentation-relevant notebooks and export them to HTML.

Run from the repo root via ``make docs-prepare``. Idempotent.

Two notebook tracks are produced:

* ``benchmark`` — passage-eval notebooks at
  ``notebooks/passage-eval/{01..03}.ipynb``. The paper-results and
  leaderboard reports read released result files only and are always
  executed. The walkthrough needs a local dataset copy; without one it is
  rendered unexecuted and flagged.

* ``generation`` — top-level dataset / solver demos at
  ``notebooks/{demo,obstacles,costmodel,solve,astar_inference_timing}.ipynb``.
  Executed on a best-effort basis: a notebook that needs data the
  current checkout doesn't have falls back to the existing outputs and
  is rendered with a "could not execute" admonition wrapper. The docs
  build never fails on a missing-data notebook.

Layout produced:

    docs/
    ├── gen/
    │   └── notebooks/
    │       ├── 01_astar_walkthrough.html
    │       ├── 02_paper_results.html
    │       ├── 03_leaderboard.html
    │       ├── demo.html
    │       ├── obstacles.html
    │       ├── costmodel.html
    │       ├── solve.html
    │       └── astar_inference_timing.html
    ├── benchmark/notebooks/
    │   ├── index.md
    │   ├── 01-astar-walkthrough.md
    │   ├── 02-paper-results.md
    │   └── 03-leaderboard.md
    └── generation/notebooks/
        ├── demo.md
        ├── obstacles.md
        ├── costmodel.md
        ├── solve.md
        └── astar_inference_timing.md
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert.exporters import HTMLExporter

REPO_ROOT = Path(__file__).resolve().parents[2]

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
def _strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


HTML_OUT = REPO_ROOT / "docs" / "gen" / "notebooks"


@dataclass(frozen=True)
class NotebookSpec:
    """One notebook + the metadata needed to render it as a docs page."""

    stem: str                              #: file stem (no extension)
    source: Path                           #: full path to the .ipynb
    section: str                           #: 'benchmark' | 'generation'
    title: str
    intro: str
    nav_label: str
    timeout_s: int = 300
    required: bool = False                 #: if True, abort on cell error
    execute_by_default: bool = True        #: opt-in default for `make docs-prepare`
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.stem.replace("_", "-").lower()

    def html_out(self) -> Path:
        return HTML_OUT / f"{self.stem}.html"

    def wrapper_out(self) -> Path:
        return REPO_ROOT / "docs" / self.section / "notebooks" / f"{self.slug}.md"


# ---------------------------------------------------------------------------
# Notebook catalogue
# ---------------------------------------------------------------------------

BENCHMARK_NOTEBOOKS: Sequence[NotebookSpec] = (
    NotebookSpec(
        stem="01_astar_walkthrough",
        source=REPO_ROOT / "notebooks" / "passage-eval" / "01_astar_walkthrough.ipynb",
        section="benchmark",
        title="Evaluating a planner, step by step",
        nav_label="Evaluator walkthrough",
        intro=(
            "Writes a planner in a few lines, scores it next to the exact A* "
            "planner on one released task, reads each of the six primary "
            "metrics, then scores both on a small validation sample. It needs "
            "the local dataset copy declared by `dataset_path`; without it, "
            "the page shows the notebook unexecuted."
        ),
    ),
    NotebookSpec(
        stem="02_paper_results",
        source=REPO_ROOT / "notebooks" / "passage-eval" / "02_paper_results.ipynb",
        section="benchmark",
        title="Paper results",
        nav_label="Paper results",
        intro=(
            "Reads the per-task results of the two paper finalists released "
            "under `src/passage-eval/results/paper-results/`. These are the "
            "historical results of the original study, reported as obtained; "
            "not re-run on the released dataset. They are not leaderboard "
            "entries."
        ),
        required=True,
    ),
    NotebookSpec(
        stem="02_score_released_baselines",
        source=REPO_ROOT / "notebooks" / "passage-eval" / "02_score_released_baselines.ipynb",
        section="benchmark",
        title="Classical baselines on the validation split",
        nav_label="Classical baselines",
        intro=(
            "Scores the three classical planners shipped with `passage-eval` "
            "(exact A*, greedy, random walk) on 64 validation tasks for each "
            "objective, with a distribution-free interval for each median "
            "regret. Reference points for reading the metrics, not benchmark "
            "results. Needs the local dataset copy."
        ),
    ),
    NotebookSpec(
        stem="03_leaderboard",
        source=REPO_ROOT / "notebooks" / "passage-eval" / "03_leaderboard.ipynb",
        section="benchmark",
        title="Recording results in the passage-eval format",
        nav_label="Results format",
        intro=(
            "Scores two classical planners, records them as `Submission` "
            "records in a temporary results folder and reads the table back: "
            "benchmark cell, six metrics, data revision and code version, "
            "without host or user names. Needs the local dataset copy."
        ),
    ),
)

GENERATION_NOTEBOOKS: Sequence[NotebookSpec] = (
    NotebookSpec(
        stem="demo",
        source=REPO_ROOT / "notebooks" / "demo.ipynb",
        section="generation",
        title="Getting started with PASSAGE",
        nav_label="Getting started",
        intro=(
            "The shortest path from the dataset to a training example: load "
            "one released scene, decode its tensor, draw a reference route "
            "and turn it into a segmentation target."
        ),
        execute_by_default=False,
    ),
    NotebookSpec(
        stem="obstacles",
        source=REPO_ROOT / "notebooks" / "obstacles.ipynb",
        section="generation",
        title="Obstacles: how much they cover and what they cost",
        nav_label="Obstacles",
        intro=(
            "Obstacle masks from least to most covered, drawn coverage "
            "against its target, and the detour that obstacles force on the "
            "optimal route over a whole validation split."
        ),
        execute_by_default=False,
    ),
    NotebookSpec(
        stem="costmodel",
        source=REPO_ROOT / "notebooks" / "costmodel.ipynb",
        section="generation",
        title="The three routing objectives",
        nav_label="The three objectives",
        intro=(
            "What the `elevation`, `slope_uphill` and `energy` objectives "
            "reward on one scene, their elevation profiles, and the extra "
            "cost of a route optimised for the wrong objective."
        ),
        execute_by_default=False,
    ),
    NotebookSpec(
        stem="solve",
        source=REPO_ROOT / "notebooks" / "solve.ipynb",
        section="generation",
        title="Solving a task with the exact reference",
        nav_label="Exact re-solve",
        intro=(
            "Solves stored labels again with the exact float64 A* reference, "
            "compares costs and routes, shows why the search budget matters, "
            "and repeats the check on a sample of validation scenes."
        ),
        execute_by_default=False,
    ),
    NotebookSpec(
        stem="astar_inference_timing",
        source=REPO_ROOT / "notebooks" / "astar_inference_timing.ipynb",
        section="generation",
        title="Timing a planner fairly",
        nav_label="Timing a planner",
        intro=(
            "Times the canonical A* planner of `passage-eval` with the "
            "practices that make a timing meaningful: warm-up, repetitions, "
            "percentiles and a stated machine."
        ),
        execute_by_default=False,
    ),
)


ALL_NOTEBOOKS = (*BENCHMARK_NOTEBOOKS, *GENERATION_NOTEBOOKS)


# ---------------------------------------------------------------------------
# Notebook execution + HTML export
# ---------------------------------------------------------------------------


def _execute(spec: NotebookSpec) -> tuple[nbformat.NotebookNode, bool, str]:
    """Best-effort execute. Returns (nb, executed, message).

    `executed=False` means we caught an exception and are falling back
    to the existing notebook contents.
    """
    nb = nbformat.read(spec.source, as_version=4)
    client = NotebookClient(
        nb,
        timeout=spec.timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(spec.source.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        if spec.required:
            raise
        clean = _strip_ansi(str(exc).splitlines()[-1])
        return nb, False, f"cell error: {clean[:120]}"
    except Exception as exc:                                   # noqa: BLE001
        if spec.required:
            raise
        clean = _strip_ansi(str(exc).splitlines()[0] if str(exc) else "")
        return nb, False, f"runtime error: {type(exc).__name__}: {clean[:120]}"
    return nb, True, "executed"


def _export_html(nb: nbformat.NotebookNode, out_path: Path) -> None:
    exporter = HTMLExporter()
    exporter.template_name = "lab"
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _resources = exporter.from_notebook_node(nb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown wrappers
# ---------------------------------------------------------------------------


def _wrapper_md(spec: NotebookSpec, *, executed: bool, message: str) -> str:
    rel_html = f"../../../gen/notebooks/{spec.stem}.html"
    rel_source = f"notebooks/{spec.source.parent.name}/{spec.source.name}" if spec.source.parent.name != "notebooks" else f"notebooks/{spec.source.name}"
    rerun = "make docs-prepare" if spec.execute_by_default else "make docs-prepare-all"
    body = f"""---
title: "{spec.title}"
hide:
  - toc
---

# {spec.title}

{spec.intro}

!!! info "Source notebook"

    `{rel_source}` — re-execute with `{rerun}` after editing.
"""
    if not executed:
        body += f"""
!!! warning "This notebook was rendered without executing"

    `{rerun}` could not execute the source notebook on this
    checkout ({message}). The HTML below shows the most recent stored
    outputs, which may be stale or empty. Re-run the notebook locally
    after fetching the data it needs (for example `make download && make
    calibrate`) to refresh.
"""
    body += f"""
<iframe
    src="{rel_html}"
    width="100%"
    height="1100"
    style="border:1px solid #eee; border-radius:6px; background:white;"
    loading="lazy">
</iframe>
"""
    return body


def _index_md(specs: Sequence[NotebookSpec], section: str) -> str:
    cards = "\n".join(
        f"""- :material-notebook-outline: **{s.nav_label}**

    ---

    {s.intro}

    [:octicons-arrow-right-24: Open the notebook](./{s.slug}.md)
"""
        for s in specs if s.section == section
    )
    title = "Benchmark notebooks" if section == "benchmark" else "Dataset / solver notebooks"
    summary = (
        "Four notebooks cover the `passage-eval` side: an evaluator "
        "walkthrough, the paper results, the classical baselines and the "
        "results format. `make docs-prepare` executes them and exports "
        "standalone HTML files embedded below; a notebook that needs a "
        "local dataset copy is rendered unexecuted and flagged."
        if section == "benchmark"
        else
        "Demo and analysis notebooks for the data-generation side. "
        "`make docs-prepare` executes the light ones; `make docs-prepare-all` "
        "also executes those that need a local dataset copy. A notebook "
        "that could not run shows its last stored outputs under a warning."
    )
    return f"""---
title: "Notebooks"
icon: material/notebook-multiple
---

# {title}

{summary}

<div class="grid cards" markdown>

{cards}

</div>
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-prepare")
    parser.add_argument("--no-execute", action="store_true",
                          help="skip in-place execution; convert as-is")
    parser.add_argument("--all", action="store_true",
                          help="force execute every notebook (overrides each\n"
                               "spec's execute_by_default flag)")
    parser.add_argument("--clean", action="store_true",
                          help="remove docs/gen/notebooks/ before regenerating")
    parser.add_argument("--only", choices=["benchmark", "generation"],
                          default=None, help="only process one track")
    args = parser.parse_args(argv)

    if args.clean and HTML_OUT.exists():
        shutil.rmtree(HTML_OUT)

    specs = [s for s in ALL_NOTEBOOKS if args.only in (None, s.section)]
    by_section: dict[str, list[NotebookSpec]] = {}
    for s in specs:
        by_section.setdefault(s.section, []).append(s)

    summary: list[str] = []
    for spec in specs:
        if not spec.source.exists():
            print(f"[docs-prepare] WARN: {spec.source} not found, skipping")
            summary.append(f"  · {spec.stem:<32} SKIP (missing source)")
            continue

        should_execute = (not args.no_execute) and (args.all or spec.execute_by_default)
        if not should_execute:
            nb = nbformat.read(spec.source, as_version=4)
            why = "--no-execute" if args.no_execute else "opt-in (run with --all or `make docs-prepare-all`)"
            executed, message = False, f"execution skipped ({why})"
        else:
            print(f"[docs-prepare] executing {spec.source.name} ...")
            # Execute in memory only: committed sources stay output-free.
            nb, executed, message = _execute(spec)

        _export_html(nb, spec.html_out())
        spec.wrapper_out().parent.mkdir(parents=True, exist_ok=True)
        spec.wrapper_out().write_text(_wrapper_md(spec, executed=executed, message=message),
                                        encoding="utf-8")
        tag = "OK " if executed else "WARN"
        summary.append(f"  · {spec.stem:<32} {tag} ({message})")
        print(f"[docs-prepare] {tag} {spec.stem}: {message}")

    # Index pages per section.
    for section, section_specs in by_section.items():
        index_path = REPO_ROOT / "docs" / section / "notebooks" / "index.md"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(_index_md(section_specs, section), encoding="utf-8")
        print(f"[docs-prepare] wrote {index_path.relative_to(REPO_ROOT)}")

    print("\n[docs-prepare] summary:")
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
