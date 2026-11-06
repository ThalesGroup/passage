---
title: Paper results
icon: material/chart-box-outline
---

# Paper results

The camera-ready PASSAGE paper compares two dense-vision planners and a
same-device CPU A* reference on all **40,000 test scenes at 256×256**, using
the `elevation` objective with weight **62.4**. This was a locked comparison
after prior exposure to the public test split, so the results are exploratory.
They are reported separately from the [leaderboard](leaderboard.md).

## What the paper measured

Both learned planners take three input channels: normalized elevation, start
and goal markers, and obstacles. Each network predicts a dense field. The
decoder sees that field, the obstacle mask and the endpoints; it has no
access to terrain values, physical edge costs or reference labels. The scorer
checks the returned route and recomputes its physical cost independently.

| Planner | Network output | Route procedure |
|---|---|---|
| Seg (path-mask segmentation) | ConvNeXt-V2 Atto predicts raw path-class logits | Weighted A* on a fixed surrogate derived from those logits |
| Reg (cost-to-go regression) | RepViT-M1 predicts a softplus cost-to-go field | Geometric Bellman descent, with goal-rooted geometric Dijkstra reconnection if descent stalls |
| Physical A* reference | No network | Exact A* on the physical edge costs |

See [Final dense-vision candidates](final-candidates.md) for decoder parameters
and checkpoint records.

On each target, the neural methods ran on the GPU, and route decoding and
physical A* ran on the CPU. Neural inference used TensorRT 10.16.2.10 on
Orin and Thor and OpenVINO 2026.4.1 on Tiger. Each persistent batch-one
session made 16 warm-up calls on one separate validation scene before timing one request per test
scene. The timer ran from raw in-memory arrays to path return; independent
host validation and scoring were separate. Neural rows average the three
within-seed statistics (seeds 1234, 2026 and 3407); the requests were not
pooled across seeds. Physical A* uses one session per target.

## Complete 256×256 test results

These are the measurements in Table 3 of the camera-ready paper. All requests
returned legal routes within budget. Cost regret values are fractions; ASSD
is normalized by reference waypoint count, and H95 is in pixels. The dashes
identify A* as the quality reference.

| Target | Planner | Median regret | p90 regret | ASSD | H95 | Success | Obstacle violations | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jetson AGX Orin | A* | — | — | — | — | 1.000 | 0.000 | 19.553 | 42.473 | 52.027 |
| Jetson AGX Orin | Seg | 0.01888 | 0.07123 | 0.03265 | 12.561 | 1.000 | 0.000 | 7.940 | 13.191 | 17.491 |
| Jetson AGX Orin | Reg | 0.07896 | 0.23422 | 0.06597 | 20.810 | 1.000 | 0.000 | 7.747 | 16.228 | 23.824 |
| Jetson AGX Thor | A* | — | — | — | — | 1.000 | 0.000 | 16.153 | 35.448 | 44.084 |
| Jetson AGX Thor | Seg | 0.01889 | 0.07127 | 0.03266 | 12.562 | 1.000 | 0.000 | 5.102 | 8.921 | 12.865 |
| Jetson AGX Thor | Reg | 0.07897 | 0.23432 | 0.06599 | 20.822 | 1.000 | 0.000 | 5.834 | 14.374 | 20.721 |
| Intel Tiger Lake-LP | A* | — | — | — | — | 1.000 | 0.000 | 12.142 | 26.162 | 29.446 |
| Intel Tiger Lake-LP | Seg | 0.01895 | 0.07140 | 0.03274 | 12.583 | 1.000 | 0.000 | 24.361 | 26.085 | 27.638 |
| Intel Tiger Lake-LP | Reg | 0.08978 | 0.25101 | 0.06716 | 21.478 | 1.000 | 0.000 | 13.950 | 19.089 | 21.145 |

Seg and Reg both had lower median latency than CPU A* on Orin and Thor.
CPU A* had the lowest median on Tiger. Seg had lower observed median and
p90 regret than Reg on all three targets. The paper documents independent
journal audits for Orin Seg seed 1234 and Thor A*, which lacked standard
completion records.

## Other resolutions and earlier validation results

The paper's latency figure uses **1,024 test scenes per grid size** outside
256×256, including at 1024×1024, and one neural seed (2026). These timings
are indicative: training budgets, numerical precision and power settings
limit comparisons across sizes. Observed returns after the 5 s deadline
remain in those timing summaries. The paper gives a full harness comparison only at 256×256.

The [historical validation notebook](notebooks/02-paper-results.md) expects
optional per-task files from earlier 256×256 validation runs. Those files
use older baseline procedures and do not reconstruct the 40,000-scene
comparison.
