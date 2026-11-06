---
title: "Leaderboard"
icon: material/trophy-outline
---

# Leaderboard

The leaderboard is the entry index for complete planning procedures on
[PASSAGE v1.0](https://huggingface.co/datasets/thalesgroup/passage), scored with
`passage-eval`. Its checked-in index is
`src/passage-eval/results/leaderboard.csv` in the source checkout.
Public rows omit host and user names.

{% include-markdown "../_partials/leaderboard_main_benchmark.md" %}

Compare the six primary metrics on matching tasks: median and p90 cost
regret, average symmetric surface distance (ASSD), percentile Hausdorff
distance, success rate and obstacle-violation rate. Compare latency separately
by hardware, runtime and timing boundary. See the [evaluation protocol](protocol.md).

## Paper results

The [paper results](candidates.md) cover the complete 40,000-scene 256×256
public test split under `elevation`. They are exploratory after prior test
exposure. Reporting a measurement in the paper does not make it a submitted
or verified leaderboard entry.

## Entry status

- **Submitted** means the submitter declared a score and its receipt.
- **Verified** means the leaderboard operator reviewed the measurement evidence
  and admitted the identified result under the applicable procedure.

A local score, an all-green test admission gate, or an
evaluator setting named `verified-candidate-run` does not grant that status.
Verification is also separate from reproducible-entry qualification or
confirmatory statistical evidence. The Hugging Face Hub's badge for
evaluations run in HF Jobs uses a separate definition.

## How to add an entry

[Evaluate your planner](contributing-a-method.md) to produce local scores and
receipts. The repository's CLI does not publish official entries, and the site
has no submission service. Maintainers can regenerate this view from the
checked-in index with `make docs-prepare`.
