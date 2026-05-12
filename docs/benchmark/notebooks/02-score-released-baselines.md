---
title: "Score released baselines from per-sample NPZ"
hide:
  - toc
---

# Score released baselines from per-sample NPZ

Replay the released vision (`seg`, `reg`) and graph (Edge-MLP) per-sample NPZ artefacts through `passage_eval.io.predictions`. Each (config × seed × host_accelerator) cell becomes one leaderboard row in `outputs/passage-eval/results/runs/r256/...`.

!!! info "Source notebook"

    `notebooks/passage-eval/02_score_released_baselines.ipynb` — re-execute with `make docs-prepare` after editing.

<iframe
    src="../../../gen/notebooks/02_score_released_baselines.html"
    width="100%"
    height="1100"
    style="border:1px solid #eee; border-radius:6px; background:white;"
    loading="lazy">
</iframe>
