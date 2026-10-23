# Paper results

This folder holds the per-task results of the two finalists reported in the
PASSAGE paper, at 256×256 with the `elevation` objective:

| Run prefix | Network | Training loss | Paper decoder |
|---|---|---|---|
| `seg-convnextv2_atto_baseline` | ConvNeXt-V2 Atto, path-mask segmentation | BCE | Dijkstra over the full grid on 1 − min-max(logit) |
| `reg-repvit_m1_bellman_vbridge` | RepViT-M1, cost-to-go regression | Huber plus a Bellman-consistency auxiliary | Strict descent, then a reduced-cost Dijkstra bridge |

Each method has three training seeds (1234, 2026 and 3407) and 1,024 validation
tasks per seed. A run directory contains `metrics.json`, with the aggregate
metrics, and `per_sample.npz`, with the per-task regret, ASSD, Hausdorff p95,
success flag, predicted cost and reference cost. Several directories repeat the
same records under different export timestamps: for a given method and seed,
their files are byte-identical. The `thor-gpu` folder name is the accelerator
label attached to these runs.

They are not leaderboard entries.
The leaderboard (`../leaderboard.csv`) starts empty; its first entries will be
the same two finalists retrained on PASSAGE v1.0 (see
[First entries](../../../../docs/benchmark/candidates.md)).
