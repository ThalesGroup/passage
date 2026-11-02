---
title: "Leaderboard"
icon: material/trophy-outline
---

# Leaderboard

The leaderboard ranks complete planning procedures on PASSAGE v1.0
([`thalesgroup/passage`](https://huggingface.co/datasets/thalesgroup/passage)),
scored with `passage-eval`. Its released index is
`src/passage-eval/results/leaderboard.csv`. Published entries carry no host or
user names.

{% include-markdown "../_partials/leaderboard_main_benchmark.md" %}

The comparison keeps the paper's six primary criteria: median and p90 cost
regret, ASSD, percentile-Hausdorff distance, success rate and obstacle-violation
rate. Geometry remains a primary criterion. Complete latency is reported
separately for each hardware and runtime. A valid-route rate is additional; it
does not replace success and obstacle violations.

## Paper results

The per-task results of the two finalists reported in the paper are released
under `src/passage-eval/results/paper-results/`, with three seeds of 1,024
validation tasks at 256² under `elevation`. They
are not leaderboard entries. Their geometry values follow the study's
normalization; see the
[alignment record](protocol.md#1-reference-paper-and-evaluation). The
[paper-results notebook](notebooks/02-paper-results.md) tabulates them.

## Entry status

Each entry will carry one of two statuses:

- **submitted**: scored by its submitter with `passage-eval` and declared with
  its receipt; not re-measured yet;
- **verified**: re-measured by Thales, once, on a qualified hardware lane.

A verified entry keeps the values its submitter declared; the re-measurement
is recorded next to them. This status is unrelated to the "verified" badge
that the Hugging Face Hub shows for evaluations run in HF Jobs.

## How to add an entry

The submission service is not deployed yet. To score a planner locally with
the `passage-eval` CLI, see [Evaluate your planner](contributing-a-method.md).
This creates local evaluation results; it does not create an official
leaderboard entry. Running `make docs-prepare` refreshes this page from the
released leaderboard.
