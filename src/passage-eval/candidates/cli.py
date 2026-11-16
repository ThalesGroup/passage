"""Operator-facing candidate commands, separate from historical paper campaigns."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile

from passage_eval.candidates.catalog import register_candidate
from passage_eval.protocol.candidate_package import load_candidate_package, read_json, artifact_path, file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passage-eval candidate")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "register"):
        p = sub.add_parser(command)
        p.add_argument("manifest", type=Path)
        if command == "register":
            p.add_argument("--catalog", required=True, type=Path)
    freeze = sub.add_parser("freeze", help="hash declared artifacts without importing code")
    freeze.add_argument("descriptor", type=Path)
    freeze.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            root = args.descriptor.resolve().parent
            if args.out.resolve().parent != root:
                raise ValueError("sealed manifest must remain in descriptor's package root")
            doc = read_json(args.descriptor)
            for item in doc["artifacts"]:
                item["sha256"] = file_sha256(artifact_path(root, item["path"]))
            descriptor, temporary = tempfile.mkstemp(prefix=".candidate-", suffix=".json", dir=root)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(doc, stream, sort_keys=True, indent=2, allow_nan=False)
                    stream.write("\n")
                load_candidate_package(temporary)
                os.link(temporary, args.out)
            finally:
                os.unlink(temporary)
            package = load_candidate_package(args.out)
        else:
            package = load_candidate_package(args.manifest)
        result = (
            register_candidate(package, args.catalog)
            if args.command == "register"
            else {"identity": package.identity, "manifest_sha256": package.digest, "status": "validated"}
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"candidate: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
