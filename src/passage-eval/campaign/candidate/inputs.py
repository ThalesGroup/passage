"""Export ordinary input arrays; keep the matching reference cohort on the host."""

from __future__ import annotations
from pathlib import Path
import numpy as np

from passage_eval.candidates.catalog import atomic_json
from passage_eval.protocol.candidate_package import read_json, file_sha256


def export_input_cohort(index_path: str | Path, output: str | Path, *, count: int | None = None) -> dict:
    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    index = read_json(index_path)
    if index.get("schema") != "passage-training-scenes/v1":
        raise ValueError("unknown prepared index schema")
    records = index["scenes"]
    count = len(records) if count is None else count
    if type(count) is not int or not 1 <= count <= len(records):
        raise ValueError("invalid cohort size")
    records = records[:count]
    size = tuple(map(int, index["dataset"]["config"].split("x")))
    if count * np.prod(size) * 9 > 2 * 1024**3 - 1024**2:
        raise ValueError("cohort exceeds input fragment bound; use smaller fragments")
    terrain, obstacles, starts, goals, ids = [], [], [], [], []
    cohort_records = []
    for entry in records:
        path = index_path.parent / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("prepared scene hash differs")
        with np.load(path, allow_pickle=False) as scene:
            z = np.asarray(scene["terrain"], dtype=np.float64)
            blocked = np.asarray(scene["obstacles"], dtype=np.bool_)
            if z.shape != size or blocked.shape != size:
                raise ValueError("prepared grid differs from declared resolution")
            terrain.append(z)
            obstacles.append(blocked)
            starts.append(np.asarray(scene["start"], dtype=np.int64))
            goals.append(np.asarray(scene["goal"], dtype=np.int64))
        ids.append(entry["sample_id"])
        cohort_records.append({**entry, "file": str(path.resolve())})
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cohort sample identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(
            stream,
            sample_ids=np.asarray(ids),
            terrain=np.asarray(terrain),
            obstacles=np.asarray(obstacles),
            starts=np.asarray(starts, dtype=np.int64),
            goals=np.asarray(goals, dtype=np.int64),
        )
    cohort_path = output.with_suffix(".cohort.json")
    atomic_json(cohort_path, {**index, "scenes": cohort_records, "source_index_sha256": file_sha256(index_path)})
    receipt = {
        "schema": "passage-input-cohort/v1",
        "n_expected": count,
        "split": index["split"],
        "source_index_sha256": file_sha256(index_path),
        "input_file_sha256": file_sha256(output),
        "cohort_index_sha256": file_sha256(cohort_path),
        "cohort_index": str(cohort_path),
    }
    atomic_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(export_input_cohort(args.index, args.output, count=args.count)))


if __name__ == "__main__":
    main()
