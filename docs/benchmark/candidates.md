---
title: First entries
icon: material/flag-checkered
---

# First entries

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

No learned-model result on PASSAGE v1.0
([`thalesgroup/passage`](https://huggingface.co/datasets/thalesgroup/passage))
is published yet. The first leaderboard entries will be the paper's two
dense-vision finalists, retrained on PASSAGE v1.0 and scored once on the test
split with `passage-eval`; their weights, decoders and frozen parameters will
be published with their entries. Thales will measure them itself, so both
will enter with the status *verified* (see
[Entry status](leaderboard.md#entry-status)).

## 1. The two finalists

Both finalists read the ordinary input tensor: elevation, start and goal
markers, and the obstacle mask. A network predicts a dense field over the grid;
a decoder then turns that field into a route on the 8-connected grid, and the
evaluator recomputes the route's cost with its own step costs.

**Table 1 — The two finalists.**

| | Segmentation finalist | Regression finalist |
|---|---|---|
| Task | Path-mask segmentation | Cost-to-go regression |
| Network | ConvNeXt-V2 Atto | RepViT-M1 |
| Training loss | BCE | Huber, plus a Bellman-consistency auxiliary term |
| Decoder in the paper | Dijkstra over the full grid on 1 − min-max(logit) | Strict descent, then a reduced-cost Dijkstra bridge |

As the [protocol](protocol.md#4-calibration-and-statistical-discipline)
requires, each entry's decoder parameters must be chosen on the calibration
split, for each grid size, and frozen before the test pass.

## 2. Paper results

The per-task results of the two finalists reported in the paper are released
under `src/passage-eval/results/paper-results/`: three training seeds of 1,024
validation tasks at 256² with the `elevation` objective. They are historical
results of the original study, reported as obtained; not re-run on the released
dataset. They are not leaderboard entries. The
[paper-results notebook](notebooks/02-paper-results.md) tabulates them.

[Evaluate your planner](contributing-a-method.md) describes how to wrap and
score any method locally. The [leaderboard page](leaderboard.md) lists the
entries; it is empty until these two finalists are scored.
