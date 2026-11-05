# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Minimal target command surface; it contains a synchronous lease wrapper, never a service manager."""

from __future__ import annotations

import argparse
from pathlib import Path

from passage_eval.io import pinned
from passage_eval.campaign.edge.package import verify_package
from passage_eval.campaign.edge.target import run_target
from passage_eval.campaign.edge.targets import validate_native_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m passage_eval.campaign.edge", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-package", allow_abbrev=False)
    verify.add_argument("--package", required=True, type=Path)
    verify.add_argument("--package-sha256", required=True)
    receipt = commands.add_parser("verify-native-adapter", allow_abbrev=False)
    receipt.add_argument("--receipt", required=True, type=Path)
    run = commands.add_parser("target-run", allow_abbrev=False)
    run.add_argument("--package", required=True, type=Path)
    run.add_argument("--package-sha256", required=True)
    run.add_argument("--receipt", required=True, type=Path)
    run.add_argument("--engine-or-model", required=True, type=Path)
    run.add_argument("--checkpoint", required=True, type=Path)
    run.add_argument("--final-decoders", required=True, type=Path)
    run.add_argument("--adapter", required=True, type=Path)
    run.add_argument("--adapter-arg", action="append", default=[])
    run.add_argument("--output", required=True, type=Path)

    def sealed_arguments(command):
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--package-sha256", required=True)
        command.add_argument("--adapter-root", required=True, type=Path)
        command.add_argument("--adapter-root-sha256", required=True)
        command.add_argument("--result-root", required=True, type=Path)
        command.add_argument("--result-root-sha256", required=True)

    leased = commands.add_parser("leased-run", allow_abbrev=False)
    leased.add_argument("--target", required=True)
    sealed_arguments(leased)
    sealed = commands.add_parser("sealed-package", allow_abbrev=False)
    sealed_arguments(sealed)
    args = parser.parse_args(argv)
    if args.command == "verify-package":
        view = verify_package(args.package, args.package_sha256)
        print(pinned.canonical_json({"package_sha256": view.sha256, "verified": True}).decode())
    elif args.command == "verify-native-adapter":
        checked = validate_native_adapter(pinned.read_json(args.receipt))
        print(pinned.canonical_json(checked).decode())
    elif args.command == "target-run":
        result = run_target(
            args.package,
            args.package_sha256,
            receipt=pinned.read_json(args.receipt),
            output=args.output,
            engine_or_model=args.engine_or_model,
            checkpoint=args.checkpoint,
            final_decoders=args.final_decoders,
            adapter_command=[str(args.adapter), *args.adapter_arg],
        )
        print(pinned.canonical_json({"target_result_sha256": result.sha256}).decode())
    else:
        from passage_eval.campaign.edge.driver import (
            EdgeJobDriver,
            SealedPackageInvocation,
            SealedRoot,
            sealed_package_argv,
        )

        invocation = SealedPackageInvocation(
            package=verify_package(args.package, args.package_sha256),
            adapter_root=SealedRoot(args.adapter_root, args.adapter_root_sha256),
            result_root=SealedRoot(args.result_root, args.result_root_sha256),
        )
        if args.command == "sealed-package":
            package = invocation.package
            sealed_package_argv(package.manifest["target"], invocation)
            print(pinned.canonical_json({"package_sha256": package.sha256, "verified": True}).decode())
        else:
            attempt = EdgeJobDriver().run_package(args.target, invocation)
            print(
                pinned.canonical_json(
                    {
                        "effect_key": attempt.effect_key,
                        "job_id": attempt.job_id,
                        "lease_retained": attempt.lease_retained,
                        "resource": attempt.resource,
                        "succeeded": attempt.succeeded,
                    }
                ).decode()
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
