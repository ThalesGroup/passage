"""Prepare immutable R256 pilot supervision from selected, hash-checked TRAIN/VAL shards."""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from passage_eval.candidates.catalog import atomic_json
from passage_eval.candidates.terrain import reverse_targets
from passage_eval.io.dataset import scene_from_row_unvalidated_label
from passage_eval.protocol.candidate_package import file_sha256


def _write_target(args):
    row, split, out, source = args
    scene = scene_from_row_unvalidated_label(row, resolution=256, cost_model="elevation", split=split, cost_weight=62.4)
    values, path = reverse_targets(scene.elevation, scene.obstacles, scene.start, scene.goal)
    if not len(path):
        raise ValueError(f"source scene {scene.sample_id} is unreachable under canonical graph")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = sha256(scene.sample_id.encode()).hexdigest()[:24] + ".npz"
    target = out / name
    if target.exists():
        raise FileExistsError(target)
    with target.open("xb") as stream:
        np.savez_compressed(
            stream,
            sample_id=np.asarray(scene.sample_id),
            terrain=scene.elevation.astype(np.float64),
            obstacles=scene.obstacles.astype(bool),
            start=np.asarray(scene.start, dtype=np.int64),
            goal=np.asarray(scene.goal, dtype=np.int64),
            expert_path=path,
            cost_to_go=values.astype(np.float32),
            reachable=np.isfinite(values),
        )
    geographic = scene.metadata.get("geographic", {})
    return {
        "sample_id": scene.sample_id,
        "file": "scenes/" + name,
        "sha256": file_sha256(target),
        "target_max": float(values[np.isfinite(values)].max()),
        "source_shard_sha256": source,
        "geographic": geographic,
    }


def prepare_split(
    source: Path, out: Path, split: str, count: int, *, seed: int, workers: int, pinned: dict[str, str]
) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError("preparation permits train and validation only")
    if workers < 1:
        raise ValueError("workers must be positive")
    files = sorted((source / "256x256" / split).glob("*.parquet"))
    counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    total = sum(counts)
    if count <= 0 or count > total:
        raise ValueError(f"requested {count} {split} scenes, available {total}")
    chosen = np.sort(np.random.default_rng(seed).choice(total, count, replace=False))
    jobs = []
    offset = 0
    source_records = []
    for file, n in zip(files, counts):
        selected = chosen[(chosen >= offset) & (chosen < offset + n)] - offset
        offset += n
        if not len(selected):
            continue
        relative = file.relative_to(source).as_posix()
        actual = file_sha256(file)
        if pinned.get(relative) != actual:
            raise ValueError(f"pinned shard hash mismatch: {relative}")
        source_records.append({"file": relative, "sha256": actual, "selected_rows": selected.tolist()})
        parquet = pq.ParquetFile(file)
        group_offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            take = selected[(selected >= group_offset) & (selected < group_offset + size)] - group_offset
            group_offset += size
            if not len(take):
                continue
            table = parquet.read_row_group(group, columns=["id", "tensor", "metadata"])
            for row in table.take(take).to_pylist():
                jobs.append((row, split, str(out / "scenes"), actual))
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "sources.json",
        {
            "files": source_records,
            "selection": "seeded sample across all shard rows",
            "seed": seed,
            "population_rows": total,
        },
    )
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(_write_target, jobs, chunksize=8), 1):
            records.append(record)
            if index % 128 == 0:
                print(json.dumps({"split": split, "prepared": index, "requested": count}), flush=True)
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate sample identity")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-count", type=int, default=32768)
    parser.add_argument("--validation-count", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("use a new immutable preparation directory")
    from huggingface_hub import HfApi

    pinned = {}
    for item in HfApi().list_repo_tree(
        "thalesgroup/passage", repo_type="dataset", revision=args.revision, recursive=True
    ):
        if getattr(item, "lfs", None) and item.path.startswith("256x256/"):
            pinned[item.path] = item.lfs.sha256
    train = prepare_split(
        args.dataset, args.out / "train", "train", args.train_count, seed=args.seed, workers=args.workers, pinned=pinned
    )
    scale = max(r["target_max"] for r in train)
    validation = prepare_split(
        args.dataset,
        args.out / "validation",
        "validation",
        args.validation_count,
        seed=args.seed + 1,
        workers=args.workers,
        pinned=pinned,
    )
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in validation}:
        raise ValueError("TRAIN/VAL identity overlap")
    dataset = {
        "id": "thalesgroup/passage",
        "revision": args.revision,
        "config": "256x256",
        "cost_model": "elevation",
        "cost_weight": 62.4,
    }
    for split, records in [("train", train), ("validation", validation)]:
        atomic_json(
            args.out / split / "index.json",
            {
                "schema": "passage-training-scenes/v1",
                "dataset": dataset,
                "split": split,
                "target_scale": scale,
                "scale_source": "train_only_max_reachable_cost_to_go",
                "seed": args.seed if split == "train" else args.seed + 1,
                "scenes": records,
            },
        )
    print(json.dumps({"train": len(train), "validation": len(validation), "target_scale": scale}), flush=True)


if __name__ == "__main__":
    main()
