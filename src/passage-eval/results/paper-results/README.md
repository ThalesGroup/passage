# Historical validation exports

This folder describes optional artifacts from earlier 256×256 validation runs
under the `elevation` objective. The camera-ready paper reports a separate
[40,000-scene public test comparison](../../../../docs/benchmark/candidates.md)
with different decoded-route procedures. These historical exports do not
reconstruct that comparison or populate the leaderboard.

| Run prefix | Network | Training loss | Historical decoder |
|---|---|---|---|
| `seg-convnextv2_atto_baseline` | ConvNeXt-V2 Atto, path-mask segmentation | BCE | Dijkstra over the full grid on 1 − min-max(logit) |
| `reg-repvit_m1_bellman_vbridge` | RepViT-M1, cost-to-go regression | Huber plus a Bellman-consistency auxiliary | Strict descent, then a reduced-cost Dijkstra bridge |

The historical study uses three training seeds (1234, 2026 and 3407) and 1,024
validation tasks per seed. When supplied, a run directory may contain
`metrics.json` with aggregate metrics and `per_sample.npz` with per-task regret,
ASSD, Hausdorff p95, success, predicted cost and reference cost. Check their
availability separately. Historical exports may repeat records under different
timestamps; verify their identities before treating them as duplicates.
The `thor-gpu` folder name is the accelerator label attached to these runs.
