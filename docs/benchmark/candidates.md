---
title: Leaderboard candidates
icon: material/flag-checkered
---

# Leaderboard candidates

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

This page records the current state of the code and weights for the two
reference candidates that will open the leaderboard. Both are the dense-vision
finalists of the paper, retrained on PASSAGE v1.0
([`thalesgroup/passage`](https://huggingface.co/datasets/thalesgroup/passage)).
Section 1 defines them and their decoders. Section 2 lists the weights for each
input resolution and the paper results. Section 3 locates the code, and
Section 4 lists what remains before the first leaderboard entries.

> **Status on 19 October 2026**
>
> No candidate below is a leaderboard entry. The submission service is not
> deployed, the test split has not been run, and the weights are not published.
> Quality figures on this page come from validation tasks; they are diagnostics.

## 1. The two candidates

Both candidates read the ordinary input tensor: elevation, start and goal
markers, and the obstacle mask. A network predicts a dense field; a decoder
then turns the field into a route on the 8-connected grid, with the physical
step costs of the evaluator.

**Table 1 — Candidate definition.**

| | Segmentation candidate | Regression candidate |
|---|---|---|
| Task | Path-mask segmentation | Cost-to-go regression |
| Network | ConvNeXt-V2 Atto | RepViT-M1 |
| Training loss | BCE | Huber, plus a Bellman-consistency auxiliary at weight 0.5 |
| Current decoder | Corridor search: A* restricted to the k % most likely free cells, dilated by r cells and widened in place until start and goal connect | A* with heuristic O + β·max(V̂ − O, 0), where O is the admissible octile bound and V̂ the predicted cost-to-go |
| Paper decoder, kept as an option | Dijkstra over the full grid on 1 − min-max(logit) | Strict descent, then a reduced-cost Dijkstra bridge |

Both decoders first apply an **output obstacle mask**: obstacle logits take
the minimum value, and the predicted cost-to-go is infinite on obstacles. On
validation tasks, the mask cost at most 1.4 % of the median request time on
CPU, and no produced route touched an obstacle.

Decoder parameters follow Rules 3 and 6 of the [protocol](protocol.md). They
are chosen on the **calibration split**, for each grid size, by a rule declared
before any calibration output: 100 % valid routes, cost-regret p90 ≤ 1 % and
p99 ≤ 5 %, then the fewest median node expansions. They are then frozen and
checked once on validation tasks that no earlier study had used.

**Table 2 — Frozen decoder parameters and their check on unseen validation tasks.**

| Grid | Segmentation (k %, r) | Regression β | Rule held: segmentation / regression |
|---|---|---|---|
| 64² | 15, 0 | 0.25 | yes / yes |
| 128² | 15, 0 | 0.25 | yes / yes |
| 256² | 5, 3 | 0.10 | yes / yes |
| 512² | 5, 0 | 0.10 | no, p99 8.0 % / yes |
| 1024² | 15, 0 | 0.10 | no, p99 9.5 % / yes |
| 2048² | 15, 0 | 0.25 | yes / no, p99 5.5 % |

Every route was valid at every grid size, and the frozen settings kept the
p90 regret below 0.32 %. The three misses concern the p99 only: with 512 tasks
or fewer, about five tasks set it. These parameters were calibrated with the
256² weights resized to each grid; they will be recalibrated for the native
weights of Section 2.

## 2. Weights by input resolution

Two generations of weights, both trained on PASSAGE v1.0, are tracked here.

- **Campaign weights.** FP32, seed 1234, 15 epochs.
  They are kept locally and identified below by the first characters of the
  checkpoint's SHA-256.
- **Current retraining.** Mixed precision (`bf16-mixed`),
  seed 1234, 15 epochs, with the paper's batch size. When that batch does not
  fit on one accelerator, gradient accumulation keeps the paper's effective
  batch.

**Table 3 — Weights by input resolution, 19 October 2026.**

| Input | Segmentation (ConvNeXt-V2 Atto + BCE) | Regression (RepViT-M1 + Bellman) |
|---|---|---|
| 64² | Retraining: batch calibration | Retraining: batch calibration |
| 128² | Campaign, epoch 15 (`e7b98307…`); retraining queued | Retraining: calibrated, queued |
| 256² | Campaign, epoch 15 (`a162b995…`); retraining: batch calibration | **Retraining running**; campaign epoch 15 *without* the Bellman term (`e06e5c96…`) |
| 512² | Retraining queued | Retraining: batch calibration |
| 1024² | Campaign stopped at epoch 13; retraining queued | Retraining: calibrated, queued |
| 2048² | **Retraining running** (about 36 h per epoch) | Not scheduled: cost-to-go targets not generated |
| 4096² | Not scheduled | Outside the paper's regression frontier |

The current decoders of Section 1 use the 256² campaign weights at every grid
size, with the input resized to 256². The regression weights `e06e5c96…` come
from the screening recipe without the Bellman term. The retrained Bellman
model will replace them.

**Paper results.** The per-task results of the two finalists reported in the
paper are released under `src/passage-eval/results/paper-results/`: three seeds
of 1,024 validation tasks at 256² with the `elevation` objective. They are the
historical results of the original study, reported as obtained; they were not
recomputed on the published dataset and are not leaderboard entries.

## 3. Where the code is

**Table 4 — Code involved in a candidate run.**

| Part | Location | State |
|---|---|---|
| Planner contract, scoring, comparison | `src/passage-eval/` (this repository) | Released with the evaluator |
| Classical baselines | `src/passage-eval/baselines/classical.py` | Released |
| Example external algorithm | `examples/external_algorithm.py` | Released; CPU only |
| Network adapters, output mask, decoders, frozen parameters | Companion vision workspace | Not yet in this repository |
| Native inference library and vendor adapters | Companion `inference` repository | CPU ONNX preparation checked; no accelerator measurement |

A candidate is therefore runnable today only from the companion workspaces.
A self-contained example in this repository would need the network adapters,
the decoders and the frozen parameters, packaged as an optional extra so that
the evaluator itself keeps no deep-learning dependency.

## 4. Before the first leaderboard entries

1. Finish the retraining of Section 2 and keep a verified copy of each
   checkpoint.
2. Recalibrate the decoders on the calibration split for each native weight,
   with a margin on the p99, and freeze the decisions under their hash.
3. Publish the weights and decoder parameters, with their license and the
   JAXA credit for the training data.
4. Run the test split once per frozen candidate with `passage-eval`.
5. Report latency only for qualified hardware and runtime profiles; each
   precision or export receives its own identity.

The [contribution guide](contributing-a-method.md) describes how to wrap and
score any method locally. The [leaderboard page](leaderboard.md) lists the
entries; it is empty until these two candidates are scored.
