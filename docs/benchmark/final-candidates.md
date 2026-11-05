---
title: Final candidates
icon: material/pin
---

# Final dense-vision candidates


## 1. Summary

This page freezes the two final candidates for the 256² evaluation and the first leaderboard entries, distinct from the historical decoders listed on the [candidates page](candidates.md): one path-mask segmentation planner (**Seg**) and one cost-to-go regression planner (**Reg**). It defines what each network outputs, how each decoder turns that output into a route, which parameters are fixed and where they were selected, and which checkpoints and export rules apply. The machine-readable copy is [`final-candidates.json`](final-candidates.json); the reference decoders are `passage_eval.baselines.final_decoders` (`src/passage-eval/baselines/final_decoders.py`), checked by `tests/passage_eval/test_final_decoders.py`.

Neither decoder reads the terrain, a reference path, a reference cost or a reachability mask. Both only receive the network output, the obstacle mask, the start and the goal. Section 2 states the common contract, Sections 3 and 4 define the candidates, Section 5 gives the current evidence and Section 6 the multi-resolution plan. The candidates aim at good, not optimal, routes: optimality is left to dedicated work.

> **Status**
>
> The 256² checkpoints below are final for seed 2026. Results quoted here come from previously exposed validation scenes and are not test results. Multi-resolution checkpoints (64², 128², 512², 1024², 2048²) are being trained; the harness test pass has not been run.

## 2. Common contract

**Input.** One NCHW float32 tensor with three channels: elevation normalised with the released calibration (min −430 m, max 8 767 m), endpoint markers (−1 everywhere, 0 at the start, 1 at the goal) and the obstacle mask. `build_input` reproduces the released tensor bit for bit.

**Graph.** Eight neighbours; a diagonal needs only a free destination. The `elevation` cost model with weight 62.4 is used **only** to score routes, never by a candidate decoder.

**Timing boundary.** A request starts from raw in-memory arrays and ends with the route array. It includes Numba pre-processing, network inference (copies and synchronisation included) and Numba post-processing. Validation and physical recosting run afterwards, outside the timer.

## 3. Seg — raw path-logit A*

**Network output.** The path-class logit \(z = \text{model}(x)[:, 1{:}2]\), taken **before** any activation: no softmax, no sigmoid.

**Decoder.** `seg_logit_astar` runs A* with step cost and heuristic

\[
q(u,v) = \ell(u,v)\,\bigl(1 + w\,(z_{\max} - z_v)\bigr), \qquad h(v) = \eta\,\operatorname{oct}(v, g),
\]

where \(\ell \in \{1, \sqrt 2\}\) and \(\operatorname{oct}\) is the octile distance. The fixed parameters are **\(w = 0.25\)** and **\(\eta = 1.5\)**. They were selected on the calibration split (256², 1 024 scenes) with a rule fixed beforehand: minimise regret p90 among settings decoding no slower than 1.5 × the (0.25, 1.5) setting.

**Checkpoint (256², seed 2026).** `eee055e0…5eb79b` (SHA-256 `eee055e084aaaec2aa4befb9db9654882d98fe629bb85e70d24a7bf5575eb79b`). Its last stage trains with BCE on the raw logit, plus 0.006 × bottleneck and 0.01 × structured path terms (AdamW, learning rate 10⁻⁶).

**Export.** The network is traced at the target resolution with static adaptive pools. For TensorRT FP16, the GRN reductions are rewritten as `Sqrt(ReduceSum(x·x))` and pinned to FP32. Without this rewrite, the outputs are entirely NaN on about half of the 512² scenes.

## 4. Reg — Bellman descent with goal-rooted reconnection

**Network output.** A softplus cost-to-go \(\hat V\), normalised by the training scale \(S = 897\,803.25\).

**Decoder.** `reg_bellman_descent` works in two steps.

1. **Descent.** From the current cell \(u\), among the free neighbours with \(\hat V(v) < \hat V(u)\) (with \(\hat V(g) = 0\)), it moves to
   \[
   v^+ = \arg\min_v \bigl[\hat V(v) + \kappa\,\tfrac{30}{S}\,\ell(u,v)\bigr].
   \]
   The added term is a **geometric** arc-cost estimate; it never reads the terrain. The fixed parameter is **\(\kappa = 8\)**, selected on the calibration split by minimising regret p90.
2. **Fallback.** If no neighbour is lower (a local minimum), a geometric Dijkstra grows from the goal over reversed edges. It stops at the first settled cell that the descent visited. The route is the descent prefix up to that cell, followed by the Dijkstra branch to the goal.

**Checkpoint (256², seed 2026).** `dc52d133…bb70` (SHA-256 `dc52d13368fd911ab44ea65801089e8d65213b6b9247dfe38116086808cabb70`). Its last stage trains with the reachable-cell reconstruction, plus 0.1 × local-policy and 0.0075 × optimal-set terms (AdamW, learning rate 10⁻⁵).

**Export.** On TensorRT, the classifier convolution, the final resize and the softplus stay FP32; the rest of the network runs FP16. An FP16 output head rounds \(\hat V\) more coarsely than neighbouring cells differ, and creates false local minima.

> **Design note — why a Bellman descent**
>
> Even on the exact cost-to-go, the plain descent \(\arg\min_v V^\star(v)\) is not optimal: it ignores the arc cost, for instance the \(\sqrt 2\) length of a diagonal step. On the calibration split it leaves 9.9 % regret p90 on a perfect map. The geometric arc term recovers most of this gap at no measurable latency cost.

## 5. Current evidence (256², exposed validation)

**Table 1 — Regret p90 / p99 and median end-to-end time, 1 024 previously exposed validation scenes.** These are exploratory, not test results. Networks run in FP16, except the Reg head on Thor (FP32 head).

| Candidate | FP32 host | Thor | Orin | Tiger | Median time Thor / Orin / Tiger |
|---|---:|---:|---:|---:|---:|
| Seg | 7.4 / 34.0 % | 7.6 / 33.3 % | 7.4 / 33.9 % | 7.4 / 36.7 % | 3.4 / 6.0 / 23.2 ms |
| Reg | 22.9 / 60.4 % | 22.9 / 58.3 % | 22.8 / 57.1 % | 24.5 / 58.0 % | 2.0 / 5.3 / 13.7 ms |

Both candidates produce valid routes on all 1 024 scenes. The physical A* reference takes 3.6–6.5 ms on the same targets. These figures are exploratory diagnostics on previously exposed validation scenes; they are not test results.

## 6. Multi-resolution plan

For 64², 128², 512², 1024² and 2048², each candidate is trained with seed 2026 on the TRAIN split of that resolution. Training starts from its 256² checkpoint and uses its own last-stage objective. A full schedule is declared up front. A first milestone checkpoint feeds the first functional and latency measurements; the same run continues to completion for the final harness evaluation. The decoder parameters stay frozen at \(w = 0.25\), \(\eta = 1.5\) and \(\kappa = 8\), and no validation or test scene enters training.
