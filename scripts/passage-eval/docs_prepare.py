"""Prepare notebook source-link pages without executing notebooks.

Run from the repo root via ``make docs-prepare``. The generated pages link
to the notebook sources; they do not embed or export notebook HTML.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class NotebookSpec:
    """One notebook + the metadata needed to render it as a docs page."""

    stem: str                              #: file stem (no extension)
    source: Path                           #: full path to the .ipynb
    section: str                           #: 'benchmark' | 'generation'
    title: str
    intro: str
    nav_label: str

    @property
    def slug(self) -> str:
        return self.stem.replace("_", "-").lower()

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
            "the notebook needs to be run after fetching the data."
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
            "under `src/passage-eval/results/paper-results/`. "
            "These are not leaderboard entries."
        ),
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
    ),
)


ALL_NOTEBOOKS = (*BENCHMARK_NOTEBOOKS, *GENERATION_NOTEBOOKS)


# ---------------------------------------------------------------------------
# Markdown wrappers
# ---------------------------------------------------------------------------


def _wrapper_md(spec: NotebookSpec) -> str:
    rel_source = spec.source.relative_to(REPO_ROOT).as_posix()
    source_url = f"https://github.com/ThalesGroup/passage/blob/main/{rel_source}"
    return f"""---
title: "{spec.title}"
hide:
  - toc
---

# {spec.title}

{spec.intro}

[Open the source notebook]({source_url})
"""


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
        "results format. Open each notebook source to view or run it locally."
        if section == "benchmark"
        else
        "Demo and analysis notebooks for the data-generation side. "
        "Open each notebook source to view or run it locally."
    )
    return f"""---
title: "Notebooks"
icon: material/notebook-multiple
---

# {title}

{summary}

[Notebook rendering and licenses](../../licenses/notebook-exports.md)

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
                        help="accepted for compatibility; notebooks are not executed")
    parser.add_argument("--all", action="store_true",
                        help="accepted for compatibility; notebooks are not executed")
    parser.add_argument("--clean", action="store_true",
                        help="accepted for compatibility; local exports are not removed")
    parser.add_argument("--only", choices=["benchmark", "generation"],
                          default=None, help="only process one track")
    args = parser.parse_args(argv)

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

        spec.wrapper_out().parent.mkdir(parents=True, exist_ok=True)
        spec.wrapper_out().write_text(_wrapper_md(spec), encoding="utf-8")
        summary.append(f"  · {spec.stem:<32} OK (source linked)")
        print(f"[docs-prepare] OK {spec.stem}: source linked")

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
