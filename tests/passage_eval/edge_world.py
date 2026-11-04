"""Small, local fixtures for the input-only edge relay tests."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


DECODER_SHA256 = "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
PLAN_SHA256 = "1" * 64
CHECKPOINT_SHA256 = "2" * 64
ENGINE_SHA256 = "3" * 64


def inputs() -> dict[str, object]:
    return {
        "sample_ids": ["validation-000"],
        "terrain": [[1.0, 2.0]],
        "obstacles": [[False, False]],
        "starts": [[0, 0]],
        "goals": [[0, 1]],
        "input_tensor": [[[1.0, 2.0]]],
    }


def package(root: Path):
    from passage_eval.campaign.edge.package import build_input_package

    return build_input_package(
        root,
        target="thor",
        plan_sha256=PLAN_SHA256,
        procedure="seg",
        inputs=inputs(),
    )


def native_artifacts(root: Path) -> dict[str, Path]:
    engine = root / "candidate.engine"
    checkpoint = root / "checkpoint.bin"
    engine.write_bytes(b"synthetic TensorRT engine")
    checkpoint.write_bytes(b"synthetic checkpoint")
    return {
        "engine_or_model": engine,
        "checkpoint": checkpoint,
        "final_decoders": Path(__file__).resolve().parents[2] / "src/passage-eval/baselines/final_decoders.py",
    }


def receipt(package_sha256: str, artifacts: dict[str, Path]) -> dict[str, str]:
    return {
        "schema": "passage-edge-native-adapter/v1",
        "target": "thor",
        "backend": "TensorRT",
        "precision": "fp16",
        "artifact_kind": "engine",
        "engine_or_model_sha256": hashlib.sha256(artifacts["engine_or_model"].read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(artifacts["checkpoint"].read_bytes()).hexdigest(),
        "procedure": "seg",
        "package_sha256": package_sha256,
        "plan_sha256": PLAN_SHA256,
        "final_decoders_sha256": DECODER_SHA256,
        "seed": 2026,
    }


def adapter(root: Path) -> list[str]:
    script = root / "synthetic-native-adapter"
    script.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--inputs', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "parser.add_argument('--engine-or-model', required=True)\n"
        "parser.add_argument('--checkpoint', required=True)\n"
        "parser.add_argument('--final-decoders', required=True)\n"
        "parser.add_argument('--receipt', required=True)\n"
        "args = parser.parse_args()\n"
        "inputs = json.loads(open(args.inputs).read())\n"
        "assert set(inputs) == {'sample_ids', 'terrain', 'obstacles', 'starts', 'goals', 'input_tensor'}\n"
        "assert '.edge-adapter-' in Path(args.inputs).parent.name\n"
        "assert all(Path(value).is_file() for value in (args.engine_or_model, args.checkpoint, args.final_decoders, args.receipt))\n"
        "assert all('.edge-adapter-' in Path(value).parent.name for value in (args.engine_or_model, args.checkpoint, args.final_decoders))\n"
        "receipt = json.loads(Path(args.receipt).read_text())\n"
        "assert receipt['engine_or_model_sha256'] and receipt['checkpoint_sha256'] and receipt['final_decoders_sha256']\n"
        "json.dump({'paths': [[[0, 0], [0, 1]]], 'timing_ms': [3.5]}, open(args.output, 'w'))\n"
    )
    script.chmod(script.stat().st_mode | os.X_OK)
    return [sys.executable, str(script)]


def mutating_adapter(root: Path) -> list[str]:
    script = root / "mutating-native-adapter"
    script.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json, os\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--inputs', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "parser.add_argument('--engine-or-model', required=True)\n"
        "parser.add_argument('--checkpoint', required=True)\n"
        "parser.add_argument('--final-decoders', required=True)\n"
        "parser.add_argument('--receipt', required=True)\n"
        "args = parser.parse_args()\n"
        "os.chmod(args.checkpoint, 0o600)\n"
        "open(args.checkpoint, 'ab').write(b'mutated')\n"
        "json.dump({'paths': [[[0, 0], [0, 1]]], 'timing_ms': [3.5]}, open(args.output, 'w'))\n"
    )
    return [sys.executable, str(script)]
