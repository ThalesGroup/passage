"""Independent host recosting and finite public projection for single-seed pilots."""

from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np

from passage_eval.candidates.catalog import atomic_json
from passage_eval.protocol.candidate_package import read_json, file_sha256, canonical_bytes
from passage_eval.protocol.contracts import PlanRequest
from passage_eval.campaign.candidate.transport import input_digest
from passage_eval.metrics.cost import path_cost
from passage_eval.metrics.summary import per_sample_metrics, aggregate_primary_metrics

METRICS = (
    "success_rate",
    "obstacle_violation_rate",
    "cost_regret_median",
    "cost_regret_p90",
    "path_assd",
    "path_hausdorff_p95",
)


def score_candidate_run(
    run_dir: str | Path, index_path: str | Path, *, target: str, gpu_seconds: float | None = None
) -> dict:
    if gpu_seconds is not None and (not math.isfinite(gpu_seconds) or gpu_seconds < 0):
        raise ValueError("training GPU cost must be nonnegative finite seconds or unknown")
    run_dir, index_path = Path(run_dir), Path(index_path)
    run = read_json(run_dir / "run.json")
    index = read_json(index_path)
    if run.get("schema") != "passage-candidate-run/v1" or index.get("split") != "validation":
        raise ValueError("pilot scoring requires a completed candidate run and validation index")
    if run["candidate"]["dataset"] != index["dataset"]:
        raise ValueError("candidate and prepared dataset identities differ")
    if sha256(canonical_bytes(run["candidate"])).hexdigest() != run["manifest_sha256"]:
        raise ValueError("candidate manifest differs from executed identity")
    if not run.get("synchronization_complete") or not run.get("runtime_close_complete"):
        raise ValueError("runtime release has not been established")
    if "container" not in run:
        raise ValueError("pilot scoring requires completed, bound container evidence")
    from passage_eval.campaign.candidate.binding import verify_container_evidence

    verify_container_evidence(run_dir, run)
    if file_sha256(run_dir / "journal.jsonl") != run["journal_sha256"]:
        raise ValueError("journal hash differs")
    journal = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text().splitlines()]
    if (
        not journal
        or journal[0].get("kind") != "session_start"
        or journal[-1].get("kind") != "session_complete"
        or journal[0].get("manifest_sha256") != run["manifest_sha256"]
    ):
        raise ValueError("session envelope or manifest identity differs")
    rows = journal[1:-1]
    expected = index["scenes"]
    if not expected:
        raise ValueError("prepared cohort must contain scenes")
    if (
        len(rows) != len(expected)
        or len(rows) != run["n_completed"]
        or len(rows) != journal[-1]["n_completed"]
        or any(
            r.get("kind") != "result" or r.get("sample_id") != entry["sample_id"] for r, entry in zip(rows, expected)
        )
    ):
        raise ValueError("journal does not exactly cover the prepared input cohort")
    paths, oracles, pred_costs, oracle_costs, obstacles, starts, goals = [], [], [], [], [], [], []
    outcomes = []
    latencies = []
    raw_violations = []
    raw_obstacle_cells = []
    objective = index["dataset"]
    for row, entry in zip(rows, expected):
        file = index_path.parent / entry["file"]
        if file_sha256(file) != entry["sha256"]:
            raise ValueError("prepared scene hash differs")
        with np.load(file, allow_pickle=False) as data:
            terrain = np.asarray(data["terrain"], dtype=np.float64)
            blocked = data["obstacles"]
            start, goal = tuple(map(int, data["start"])), tuple(map(int, data["goal"]))
            oracle = data["expert_path"]
        request = PlanRequest(
            entry["sample_id"], terrain, blocked, start, goal, objective["cost_model"], objective["cost_weight"]
        )
        if input_digest(request) != row["raw_input_sha256"]:
            raise ValueError("executed input identity differs")
        status = row["status"]
        from passage_eval.protocol.contracts import STATUSES

        if status not in STATUSES:
            raise ValueError("unknown terminal status")
        raw = np.asarray(row["path"])
        physical = path_cost(terrain, raw, weight=objective["cost_weight"], cost_model=objective["cost_model"])
        from passage_eval.metrics.reliability import path_status

        _, violated, n_blocked = path_status(raw, start=start, goal=goal, obstacles=blocked)
        raw_violations.append(violated)
        raw_obstacle_cells.append(n_blocked)
        # A late/failed legal route cannot turn into success when rescored.
        predicted = raw if status == "ok" else []
        paths.append(predicted)
        oracles.append(oracle)
        pred_costs.append(physical)
        oracle_costs.append(
            path_cost(terrain, oracle, weight=objective["cost_weight"], cost_model=objective["cost_model"])
        )
        obstacles.append(blocked)
        starts.append(start)
        goals.append(goal)
        latencies.append(row["elapsed_ns"] / 1e6)
        outcomes.append(status)
    arrays = per_sample_metrics(
        predicted_paths=paths,
        oracle_paths=oracles,
        predicted_costs=pred_costs,
        oracle_costs=oracle_costs,
        obstacles=obstacles,
        starts=starts,
        goals=goals,
    )
    arrays["obstacle_violation"] = np.asarray(raw_violations, dtype=bool)
    arrays["obstacle_cells"] = np.asarray(raw_obstacle_cells, dtype=np.int64)
    # The evaluator independently corrects an invalid candidate 'ok' report.
    for i, valid in enumerate(arrays["valid_path"]):
        if outcomes[i] == "ok" and not valid:
            outcomes[i] = "invalid_path"
    raw_metrics = aggregate_primary_metrics(arrays)
    metrics = {k: float(raw_metrics[k]) if np.isfinite(raw_metrics[k]) else None for k in METRICS}
    finite = [v for r, v in zip(rows, latencies) if not r.get("latency_censored", False)]
    censored = len(rows) - len(finite)
    quantiles = {}
    for label, q in [("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99)]:
        rank = int(np.ceil(q * len(rows)))
        quantiles[label] = None if rank > len(finite) else float(sorted(finite)[rank - 1])
    np.savez_compressed(
        run_dir / "scored-arrays.npz",
        **arrays,
        sample_ids=np.asarray([e["sample_id"] for e in expected]),
        elapsed_ms=np.asarray(latencies),
        statuses=np.asarray(outcomes),
    )
    counts = dict(Counter(outcomes))
    peaks = {}
    for result in rows:
        for key, value in result.get("resource_peaks", {}).items():
            peaks[key] = max(peaks.get(key, 0), value)
    atomic_json(
        run_dir / "score.json",
        {
            "schema": "passage-pilot-score/v1",
            "metrics": metrics,
            "status_counts": counts,
            "terminal_stage_counts": dict(Counter(result.get("stage") or "unknown" for result in rows)),
            "precision_counts": dict(Counter(result.get("precision", "unspecified") for result in rows)),
            "reported_resource_peaks": peaks,
            "run_sha256": file_sha256(run_dir / "run.json"),
            "input_index_sha256": file_sha256(index_path),
        },
    )
    candidate = run["candidate"]
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_label": candidate["candidate_id"],
        "family": candidate["family"],
        "version": candidate["version"],
        "seed": candidate["seed"],
        "target": target,
        "execution": "gpu" if run["device"] == "cuda" else "cpu",
        "profile_id": "pilot-validation-r256/v1",
        "n_expected": len(rows),
        "benchmark": {
            "dataset_id": objective["id"],
            "dataset_revision": objective["revision"],
            "config": objective["config"],
            "resolution": int(objective["config"].split("x")[0]),
            "split": "validation",
        },
        "metrics": metrics,
        "latency": {**quantiles, "n_censored": censored, "boundary": run["timing_boundary"]},
        "status_counts": counts,
        "training": {
            "seed": candidate["seed"],
            "gpu_seconds": gpu_seconds,
            "selection": "reference" if candidate["seed"] is None else "exploratory_single_seed",
        },
        "evidence": {
            "run_sha256": file_sha256(run_dir / "run.json"),
            "manifest_sha256": run["manifest_sha256"],
            "input_index_sha256": file_sha256(index_path),
            "score_sha256": file_sha256(run_dir / "score.json"),
        },
        "status": "measured_unadmitted",
        "verification": None,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu-seconds", type=float, default=None, help="actual training GPU seconds; omit when unknown"
    )
    args = parser.parse_args(argv)
    row = score_candidate_run(args.run, args.index, target=args.target, gpu_seconds=args.gpu_seconds)
    atomic_json(args.output, row)
    print(json.dumps(row["metrics"]))


if __name__ == "__main__":
    main()
