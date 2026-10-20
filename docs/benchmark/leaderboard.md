---
title: "Leaderboard"
icon: material/trophy-outline
---

# Leaderboard

The leaderboard ranks complete planning procedures on PASSAGE v1.0
([`thalesgroup/passage`](https://huggingface.co/datasets/thalesgroup/passage)),
scored with `passage-eval`. Its released index is
`src/passage-eval/results/leaderboard.csv`.

{% include-markdown "../_partials/leaderboard_main_benchmark.md" %}

The comparison retains the paper's six primary criteria: median and p90 cost
regret, ASSD, percentile-Hausdorff distance, success rate and obstacle-violation
rate. Geometry remains a primary criterion. Complete latency is reported
separately for each qualified hardware/runtime profile. A valid-route rate is
additional; it does not replace success and obstacle violations.

## Paper results

The per-task results of the two finalists reported in the paper are released
under `src/passage-eval/results/paper-results/`, with three seeds of 1,024
validation tasks at 256² under `elevation`. They are the historical results of
the original study, reported as obtained: they were not recomputed on the
published dataset and are not leaderboard entries. Their geometry values follow
the study's normalization; see the
[alignment record](protocol.md#1-reference-paper-and-alignment). The
[paper-results notebook](notebooks/02-paper-results.md) tabulates them.

## How to add an entry

Prepare and score a local candidate with the `passage-eval` CLI — see
[Contributing a method](contributing-a-method.md). This creates local evaluation
artifacts; it does not submit an official leaderboard entry. The submission
service is not deployed yet. Running `make docs-prepare` refreshes this page
from the released leaderboard.
