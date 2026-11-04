#!/usr/bin/env python
"""Build the label-free VALIDATION rehearsal roster of the PASSAGE leaderboard chain.

This is a thin wrapper around ``scripts/build_public_test_roster.py``: it reuses
``build_roster`` and ``write_outputs`` unchanged, so the roster has the same schema
(``passage-roster-v2``), selection rule, input identity spec, geographic block
fields (including the footprint requirement, on unless ``--allow-missing-footprints``)
and label-free column allowlist as the public-test roster. It differs in three
ways only:

* the split is fixed to ``validation`` and the dataset root must be a pinned
  snapshot of the validation split (its ``PINNED.json`` declares
  ``subset.split == "validation"`` and lists no other split); the test split is
  refused, so this script cannot read it;
* the benchmark id defaults to ``passage-validation-rehearsal-v1`` and the
  ``dataset`` block says that the roster is a rehearsal, not the public test
  benchmark (the digest is recomputed after that change and validated);
* the default roster size is 64 tasks.

Selection rule (inherited): rank every row id of the split by
``sha256((salt + id).encode("utf-8")).hexdigest()`` ascending and keep the first
``n``. The default salt is ``<benchmark_id>/r<resolution:04d>/roster-v1/``, i.e.
``passage-validation-rehearsal-v1/r0256/roster-v1/``.

Example::

    nice -n 10 uv run --frozen python scripts/build_rehearsal_roster.py \\
        --dataset-root /path/to/validation-dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "passage-validation-rehearsal-v1"
SPLIT = "validation"
PUBLIC_TEST_BENCHMARK_ID = "passage-public-test-v1"
PURPOSE = (
    "end-to-end rehearsal of the leaderboard chain on the validation split; "
    "not the public test benchmark and not a result-bearing roster"
)
VISIBILITY = (
    "validation split of the pinned Hub commit; rehearsal only, not the public test benchmark; "
    "validation labels are readable by the scorer-side loader"
)
CAVEAT = (
    "Validation scenes may overlap scenes exposed during candidate development: the candidates were "
    "developed on a local successor copy of the dataset and its validation bytes were not compared "
    "with this pinned snapshot by this script. Do not report rehearsal numbers as held-out results."
)


def _load_base():
    """Import the existing builder by path (``scripts/`` is not a package; ``-I`` drops it from sys.path)."""
    path = REPOSITORY / "scripts" / "build_public_test_roster.py"
    spec = importlib.util.spec_from_file_location("build_public_test_roster", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_validation_root(root: Path, *, config: str = "256x256") -> dict[str, Any]:
    """Refuse any dataset root that is not a pinned snapshot of the validation split only."""
    receipt = json.loads((root / "PINNED.json").read_bytes().decode("utf-8"))
    subset = receipt.get("subset") or {}
    if subset.get("split") != SPLIT or subset.get("config") != config:
        raise ValueError(
            f"{root}/PINNED.json must declare subset {config}/{SPLIT}, not {subset.get('config')}/{subset.get('split')}"
        )
    prefix = f"{config}/{SPLIT}/"
    foreign = sorted(f["path"] for f in receipt.get("files", []) if not f["path"].startswith(prefix))
    if foreign or not receipt.get("files"):
        raise ValueError(f"{root}/PINNED.json lists files outside {prefix}: {foreign[:3]}")
    if (root / config / "test").exists():
        raise ValueError(
            f"{root} holds a {config}/test directory; the rehearsal root must contain the validation split only"
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-root", required=True, type=Path, help="pinned VALIDATION snapshot root holding PINNED.json"
    )
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "outputs" / "rosters")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cost-model", default="elevation")
    parser.add_argument("--cost-weight", type=float, default=62.4)
    parser.add_argument("--task-variant", default="obstacles", choices=["obstacles", "free"])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--salt", default=None)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument(
        "--allow-missing-footprints",
        action="store_true",
        help="build a rehearsal roster without planned block analysis (tasks without a footprint stay unplaced)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.benchmark_id == PUBLIC_TEST_BENCHMARK_ID or "test" in args.benchmark_id.split("-"):
        raise SystemExit(
            f"Refusing benchmark id {args.benchmark_id!r}: this script builds validation rehearsal rosters only"
        )

    base = _load_base()
    config = f"{args.resolution}x{args.resolution}"
    root = args.dataset_root.resolve()
    check_validation_root(root, config=config)
    base.COLUMNS_REQUESTED.clear()
    roster, info = base.build_roster(
        root,
        benchmark_id=args.benchmark_id,
        resolution=args.resolution,
        split=SPLIT,
        cost_model=args.cost_model,
        cost_weight=args.cost_weight,
        task_variant=args.task_variant,
        n_expected=args.n,
        salt=args.salt,
        verify_hashes=not args.skip_file_hashes,
        require_footprints=not args.allow_missing_footprints,
    )
    if roster["dataset"]["split"] != SPLIT or any(
        not t["source_file"].startswith(f"{config}/{SPLIT}/") for t in roster["tasks"]
    ):
        raise ValueError("The built roster references something other than the validation split")
    # Say plainly what this roster is; the digest covers these fields, so recompute and re-validate it.
    roster["dataset"]["visibility"] = VISIBILITY
    roster["dataset"]["purpose"] = PURPOSE
    roster["roster_sha256"] = base.compute_roster_sha256(roster)
    base.validate_roster(roster)

    own_argv = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    paths = base.write_outputs(
        roster,
        info,
        args.out_dir,
        argv=own_argv,
        dataset_root=root,
        verify_hashes=not args.skip_file_hashes,
        overwrite=args.overwrite,
    )
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["rehearsal"] = {
        "purpose": PURPOSE,
        "split": SPLIT,
        "benchmark_id": roster["benchmark_id"],
        "is_public_test_benchmark": False,
        "wrapper_script": {"path": str(Path(__file__).resolve()), "sha256": base.file_sha256(Path(__file__).resolve())},
        "wraps": "scripts/build_public_test_roster.py build_roster and write_outputs, unchanged; the dataset block's "
        "visibility and purpose were rewritten and roster_sha256 recomputed and validated afterwards",
        "validation_labels_read_by_build": False,
        "development_overlap_caveat": CAVEAT,
    }
    paths["receipt"].write_text(json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "roster_sha256": roster["roster_sha256"],
                "paths": {k: str(v) for k, v in paths.items()},
                "geographic_blocks": info["block_statistics"],
                "footprint_status": info["footprint_status"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
