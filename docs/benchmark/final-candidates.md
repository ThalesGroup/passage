---
title: Final candidates
icon: material/pin
---

# Final dense-vision candidates


## 1. Summary

This page describes the frozen 256×256 decoder settings and the seed-2026
checkpoints for the two dense-vision candidates: path-mask segmentation
(**Seg**) and cost-to-go regression (**Reg**). The machine-readable records
are [the candidate manifest](final-candidates.json) and
[the replica catalogue for seeds 1234 and 3407](replica-candidates-256.json).
The reference decoders live in
`passage_eval.baselines.final_decoders`.

Both decoders use only the network output, the obstacle mask, the start and
the goal. The [paper results](candidates.md) describe the locked comparison on
all 40,000 public 256×256 test scenes. The checkpoint and validation records
below describe separate evidence and do not establish a leaderboard entry.

## 2. Common contract

**Input.** One NCHW float32 tensor with three channels: elevation normalised with the released calibration (min −430 m, max 8 767 m), endpoint markers (−1 everywhere, 0 at the start, 1 at the goal) and the obstacle mask. `build_input` reproduces the released tensor bit for bit.

**Graph.** Eight neighbours; a diagonal needs only a free destination. The `elevation` cost model with weight 62.4 is used **only** to score routes, never by a candidate decoder.

**Timing boundary.** A request starts from raw in-memory arrays and ends with the route array. For the direct native boundary, this includes input preparation, device
transfers, inference, synchronization and route decoding with any fallback.
Independent host validation and physical recosting run outside the timer.

## 3. Seg: weighted A* on raw path logits

**Network output.** The path-class logit \(z = \text{model}(x)[:, 1{:}2]\), taken **before** any activation: no softmax, no sigmoid.

**Decoder.** `seg_logit_astar` runs A* with step cost and heuristic

\[
q(u,v) = \ell(u,v)\,\bigl(1 + w\,(z_{\max} - z_v)\bigr), \qquad h(v) = \eta\,\operatorname{oct}(v, g),
\]

where \(\ell \in \{1, \sqrt 2\}\) and \(\operatorname{oct}\) is the octile distance. The fixed parameters are **\(w = 0.25\)** and **\(\eta = 1.5\)**. They were selected on the calibration split (256², 1 024 scenes) with a rule fixed beforehand: minimise regret p90 among settings decoding no slower than 1.5 × the (0.25, 1.5) setting.

**Checkpoint (256², seed 2026).** `eee055e0…5eb79b` (SHA-256 `eee055e084aaaec2aa4befb9db9654882d98fe629bb85e70d24a7bf5575eb79b`). Its last stage trains with BCE on the raw logit, plus 0.006 × bottleneck and 0.01 × structured path terms (AdamW, learning rate 10⁻⁶).

**Export.** The network is traced at the target resolution with static adaptive pools. For TensorRT FP16, the GRN reductions are rewritten as `Sqrt(ReduceSum(x·x))` and pinned to FP32. The frozen export record notes that the original FP16 reductions could produce non-finite output at 512×512.

## 4. Reg: geometric Bellman descent with goal-rooted reconnection

**Network output.** A softplus cost-to-go \(\hat V\), normalised by the training scale \(S = 897\,803.25\).

**Decoder.** `reg_bellman_descent` works in two steps.

1. **Descent.** From the current cell \(u\), among the free neighbours with \(\hat V(v) < \hat V(u)\) (with \(\hat V(g) = 0\)), it moves to
   \[
   v^+ = \arg\min_v \bigl[\hat V(v) + \kappa\,\tfrac{30}{S}\,\ell(u,v)\bigr].
   \]
   The added term is a **geometric** arc-cost estimate; it never reads the terrain. The fixed parameter is **\(\kappa = 8\)**, selected on the calibration split by minimising regret p90.
2. **Fallback.** If no neighbour is lower (a local minimum), a geometric Dijkstra grows from the goal over reversed edges. It stops at the first settled cell that the descent visited. The route is the descent prefix up to that cell, followed by the Dijkstra branch to the goal.

**Checkpoint (256², seed 2026).** `dc52d133…bb70` (SHA-256 `dc52d13368fd911ab44ea65801089e8d65213b6b9247dfe38116086808cabb70`). Its last stage trains with reachable-cell MSE plus 0.5 × Bellman (Smooth-L1), 0.1 × local-policy and 0.00747 × optimal-set losses (AdamW, learning rate 10⁻⁵).

**Export.** On TensorRT, the classifier convolution, the final resize and the softplus stay FP32; the rest of the network runs FP16. An FP16 output head rounds \(\hat V\) more coarsely than neighbouring cells differ, and creates false local minima.

> **Design note — why a Bellman descent**
>
> Even on the exact cost-to-go, the plain descent \(\arg\min_v V^\star(v)\) is not optimal: it ignores the arc cost, for instance the \(\sqrt 2\) length of a diagonal step. The decoder adds a geometric approximation to the missing arc cost before it chooses the next cell.

## 5. Historical validation measurements

**Table 1 — Regret p90 / p99 and median end-to-end time at 256×256, 1,024 previously exposed validation scenes.** These are exploratory, not test results. Networks run in FP16, except the Reg head on Thor (FP32 head).

| Candidate | FP32 host | Thor | Orin | Tiger | Median time Thor / Orin / Tiger |
|---|---:|---:|---:|---:|---:|
| Seg | 7.4 / 34.0 % | 7.6 / 33.3 % | 7.4 / 33.9 % | 7.4 / 36.7 % | 3.4 / 6.0 / 23.2 ms |
| Reg | 22.9 / 60.4 % | 22.9 / 58.3 % | 22.8 / 57.1 % | 24.5 / 58.0 % | 2.0 / 5.3 / 13.7 ms |

Both candidates produced valid routes on all 1,024 scenes in this validation
cohort. The physical A* reference took 3.6–6.5 ms on the same targets in this
study.

## 6. Other resolutions

The candidate manifest preserves an earlier training plan. The camera-ready
paper reports the procedures actually measured: outside 256×256, each grid
size uses 1,024 public test scenes and one neural seed (2026), with
preprocessing and decoder settings fixed from 256×256. Training budgets and
other conditions differ by size. See [the paper results](candidates.md#other-resolutions-and-earlier-validation-results)
for the limits of these indicative timings.
