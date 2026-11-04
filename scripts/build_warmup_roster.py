#!/usr/bin/env python
"""Build the label-free warm-up roster of a timed pass: validation scenes disjoint from the evaluated roster.

The warm-up inputs are the first ``k`` validation row ids ranked by
``sha256((salt + id).encode("utf-8")).hexdigest()`` that are not evaluated
tasks, by sample id and by input content identity (the input identity without
its ordinary-metadata digest; ``passage_eval.protocol.warmup.select_warmup_scenes``).
The rows come from the
existing label-free builder (``scripts/build_public_test_roster.py``
``build_roster``) with the split fixed to ``validation``; the cell (resolution,
cost model, weight, task variant) is the evaluated roster's. The output is a
``passage-roster-v2`` the supervisor accepts as ``--warmup-roster``, plus a
receipt with the selection record.

Proposed, not sealed: eight scenes is the protocol's proposal (``--k``); the
salt and the dataset root have no default and belong to the protocol author.

By default the dataset root must be a pinned snapshot of the validation split
only. ``--allow-shared-root`` accepts a pinned root whose ``PINNED.json`` also
lists other splits, so that the warm-up roster and the evaluated roster pin the
same root; only files under ``<N>x<N>/validation/`` are listed, verified and read.

Example::

    nice -n 10 uv run --frozen python scripts/build_warmup_roster.py \\
        --dataset-root /path/to/validation-dataset --salt <sealed salt> \\
        --evaluated-roster rosters/<evaluated>.json --evaluated-roster-sha256 <pinned digest>
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "passage-validation-warmup-v1"
SPLIT = "validation"


def _load_base():
    """Import the existing builder by path (``scripts/`` is not a package; ``-I`` drops it from sys.path)."""
    path = REPOSITORY / "scripts" / "build_public_test_roster.py"
    spec = importlib.util.spec_from_file_location("build_public_test_roster", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_root(root: Path, *, config: str, shared: bool) -> dict[str, Any]:
    """Refuse a root that is not a pinned validation snapshot (or, with ``shared``, that lists no validation file)."""
    receipt = json.loads((root / "PINNED.json").read_bytes().decode("utf-8"))
    prefix = f"{config}/{SPLIT}/"
    files = receipt.get("files") or []
    if not any(entry.get("path", "").startswith(prefix) for entry in files):
        raise ValueError(f"{root}/PINNED.json lists no file under {prefix}")
    if shared:
        return receipt
    subset = receipt.get("subset") or {}
    if subset.get("split") != SPLIT or subset.get("config") != config:
        raise ValueError(
            f"{root}/PINNED.json must declare subset {config}/{SPLIT} (or pass --allow-shared-root), "
            f"not {subset.get('config')}/{subset.get('split')}"
        )
    foreign = sorted(entry["path"] for entry in files if not entry["path"].startswith(prefix))
    if foreign:
        raise ValueError(
            f"{root}/PINNED.json lists files outside {prefix} (or pass --allow-shared-root): {foreign[:3]}"
        )
    if (root / config / "test").exists():
        raise ValueError(f"{root} holds a {config}/test directory (or pass --allow-shared-root)")
    return receipt


def main(argv: list[str] | None = None) -> int:
    from passage_eval.io.roster import load_roster
    from passage_eval.protocol.guards import add_test_admission_arguments, test_split_allowed
    from passage_eval.protocol.warmup import PROPOSED_WARMUP_COUNT, build_warmup_roster

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", required=True, type=Path, help="pinned root holding PINNED.json")
    parser.add_argument("--salt", required=True, help="the sealed warm-up salt (no default)")
    parser.add_argument("--evaluated-roster", required=True, type=Path, help="the roster the timed pass evaluates")
    parser.add_argument("--evaluated-roster-sha256", required=True, metavar="HEX")
    parser.add_argument("--k", type=int, default=PROPOSED_WARMUP_COUNT, help="warm-up inputs (proposed: 8)")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "outputs" / "rosters")
    parser.add_argument("--allow-shared-root", action="store_true", help="the root's PINNED.json lists other splits")
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    add_test_admission_arguments(parser, what="an evaluated roster of the test split", preparation=True)
    args = parser.parse_args(argv)
    admission = test_split_allowed(
        args.allow_test_split,
        args.test_admission_record,
        evidence_dir=args.test_admission_evidence_dir,
        preparation_approval=args.test_preparation_approval,
    )
    if "test" in args.benchmark_id.lower().replace("_", "-").split("-"):
        raise SystemExit(f"Refusing benchmark id {args.benchmark_id!r}: warm-up rosters are validation rosters")

    base = _load_base()
    evaluated = load_roster(args.evaluated_roster, expected_sha256=args.evaluated_roster_sha256)
    cell, resolution = evaluated["cell"], evaluated["dataset"]["resolution"]
    config = f"{resolution}x{resolution}"
    root = args.dataset_root.resolve()
    check_root(root, config=config, shared=args.allow_shared_root)
    base.COLUMNS_REQUESTED.clear()
    calls = {"n": 0}

    def build_pool(n: int) -> dict[str, Any]:
        calls["n"] += 1
        roster, _ = base.build_roster(
            root,
            benchmark_id=args.benchmark_id,
            resolution=resolution,
            split=SPLIT,
            cost_model=cell["cost_model"],
            cost_weight=cell["cost_weight"],
            task_variant=cell["task_variant"],
            n_expected=n,
            salt=args.salt,
            verify_hashes=not args.skip_file_hashes and calls["n"] == 1,  # sizes are still checked on later calls
            require_footprints=False,  # warm-up inputs are never analysed by blocks
        )
        return roster

    roster, selection = build_warmup_roster(
        build_pool,
        salt=args.salt,
        evaluated_roster=evaluated,
        benchmark_id=args.benchmark_id,
        k=args.k,
        allow_test=admission is not None,
        admission=admission,
    )
    stem = f"{roster['benchmark_id']}-r{resolution:04d}-warmup-{roster['schema'].removeprefix('passage-')}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "roster": args.out_dir / f"{stem}.json",
        "sidecar": args.out_dir / f"{stem}.sha256",
        "receipt": args.out_dir / f"{stem}.receipt.json",
    }
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite an existing warm-up roster in {args.out_dir}; pass --overwrite")
    paths["roster"].write_text(json.dumps(roster, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    paths["sidecar"].write_text(roster["roster_sha256"] + "\n", encoding="ascii")
    receipt = {
        "schema": "passage-warmup-roster-receipt/v1",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roster_sha256": roster["roster_sha256"],
        "roster_file_sha256": base.file_sha256(paths["roster"]),
        "selection": selection,
        "evaluated_roster_sha256": evaluated["roster_sha256"],
        "pinned_receipt_sha256": roster["dataset_root_identity"]["sha256"],
        "shared_root": args.allow_shared_root,
        "file_verification": "sha256 of every validation file recomputed once"
        if not args.skip_file_hashes
        else "size only (--skip-file-hashes)",
        "label_policy": {"columns_requested": sorted(base.COLUMNS_REQUESTED), "not_read": base.NOT_READ},
        "code": {
            name: base.file_sha256(path)
            for name, path in {
                "warmup_script": Path(__file__).resolve(),
                "build_script": REPOSITORY / "scripts" / "build_public_test_roster.py",
            }.items()
        },
        "decision_status": selection["decision_status"],
    }
    paths["receipt"].write_text(json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "roster_sha256": roster["roster_sha256"],
                "selected_ids": selection["selected_ids"],
                "selection_sha256": selection["selection_sha256"],
                "paths": {key: str(value) for key, value in paths.items()},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
