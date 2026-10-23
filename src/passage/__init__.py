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

"""PASSAGE: Path Solving Dataset and Generator."""

import argparse
import sys

from loguru import logger

# Lazily import heavy CLI functions to avoid importing dependencies (like rasterio) at package import time.
# Tests patch these names (e.g., patch('passage.run_generate')) so keep attributes available as placeholders.
run_calibrate = None
run_download = None
run_generate = None
run_regenerate = None
compress_release = None
prepare_export = None
upload_export = None
export_and_upload = None


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def main(argv: list[str] | None = None) -> None:
    """Entry point for the PASSAGE CLI."""
    parser = argparse.ArgumentParser(
        prog="passage",
        description="PASSAGE: Path Solving Dataset and Generator",
    )

    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download",
        help="Download ALOS tiles and track URL status",
    )
    download_parser.add_argument(
        "--outputs-dirpath",
        default="outputs/download",
        help="Directory for output files (default: outputs/download)",
    )
    download_parser.add_argument(
        "--archive",
        type=int,
        nargs=2,
        action="append",
        metavar=("LAT", "LON"),
        help="Download only these geographic archives (repeat for each latitude/longitude pair)",
    )
    download_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from existing download.csv",
    )
    download_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for multiprocessing (0 disables multiprocessing, default: all CPUs)",
    )
    download_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate dataset samples from DSM tiles",
    )
    generate_parser.add_argument(
        "--outputs-dirpath",
        default="outputs/dataset",
        help="Directory for output files (default: outputs/dataset)",
    )
    generate_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml file (default: config/config.yaml)",
    )
    generate_parser.add_argument(
        "--tiles-dirpath",
        default="outputs/download/tiles",
        help="Directory containing DSM tiles (default: outputs/download/tiles)",
    )
    generate_parser.add_argument(
        "--calibration-filepath",
        default="outputs/calibrate/calibrate.json",
        help="Path to calibration JSON file (default: outputs/calibrate/calibrate.json)",
    )
    generate_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Image resolution in pixels (overrides config). Provide one or more space-separated values, e.g. --resolution 64 128",
    )
    generate_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate (overrides config)",
    )
    generate_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for multiprocessing (can be specified before or after subcommand)",
    )
    generate_parser.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="Disable saving PNG images (saves tensors only)",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete existing directories and regenerate all samples (default: resume from existing)",
    )
    generate_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Find min/max elevation values across all DSM tiles",
    )
    calibrate_parser.add_argument(
        "--outputs-dirpath",
        default="outputs/calibrate",
        help="Directory for output files (default: outputs/calibrate)",
    )
    calibrate_parser.add_argument(
        "--tiles-dirpath",
        default="outputs/download/tiles",
        help="Directory containing DSM tiles (default: outputs/download/tiles)",
    )
    calibrate_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing calibrate.json (recompute all tiles and overwrite existing results)",
    )
    calibrate_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for multiprocessing (can be specified before or after subcommand)",
    )
    calibrate_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    compress_parser = subparsers.add_parser(
        "compress",
        help="Generate parquet shards for the selected resolutions",
    )
    compress_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml file (default: config/config.yaml)",
    )
    compress_parser.add_argument(
        "--dataset-dirpath",
        default="outputs/dataset",
        help="Directory containing generated datasets (default: outputs/dataset)",
    )
    compress_parser.add_argument(
        "--calibrate-dirpath",
        default="outputs/calibrate",
        help="Directory used to derive the export directory (default: outputs/calibrate)",
    )
    compress_parser.add_argument(
        "--calibration-filepath",
        default=None,
        help="Calibration JSON used to derive the export directory (takes precedence over --calibrate-dirpath)",
    )
    compress_parser.add_argument(
        "--export-dirpath",
        default=None,
        help="Explicit export directory override (default: derive from calibration path)",
    )
    compress_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Specific resolutions to compress. If not specified, compress all dataset resolutions.",
    )
    compress_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for multiprocessing (can be specified before or after subcommand)",
    )
    compress_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Prepare the export directory for upload by refreshing notebooks, assets, calibration, and README",
    )
    export_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml file (default: config/config.yaml)",
    )
    export_parser.add_argument(
        "--dataset-dirpath",
        default="outputs/dataset",
        help="Directory containing generated datasets (default: outputs/dataset)",
    )
    export_parser.add_argument(
        "--calibrate-dirpath",
        default="outputs/calibrate",
        help="Directory containing calibration data (default: outputs/calibrate)",
    )
    export_parser.add_argument(
        "--calibration-filepath",
        default=None,
        help="Path to calibration JSON file (default: <calibrate-dirpath>/calibrate.json; takes precedence over --calibrate-dirpath)",
    )
    export_parser.add_argument(
        "--notebooks-dirpath",
        default="notebooks",
        help="Directory containing notebooks (default: notebooks)",
    )
    export_parser.add_argument(
        "--export-dirpath",
        default=None,
        help="Explicit export directory override (default: derive from calibration path)",
    )
    export_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Specific resolutions to refresh in the export directory. If not specified, use prepared configs.",
    )
    export_parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id to embed in the export README (default: thalesgroup/passage)",
    )
    export_parser.add_argument(
        "--keep-existing-configs",
        action="store_true",
        help="Retain compatible configs from the existing export README after verifying their earlier remote uploads",
    )
    export_parser.add_argument(
        "--max-notebook-retries",
        type=_positive_int,
        default=3,
        help="Finite number of notebook execution attempts before giving up (default: 3)",
    )
    export_parser.add_argument(
        "--notebook-retry-delay",
        type=int,
        default=5,
        help="Seconds to wait between notebook execution retries (default: 5)",
    )
    export_parser.add_argument(
        "--force",
        action="store_true",
        help="Force every step regardless of existing artifacts (overrides per-step skip-if-existing)",
    )
    export_parser.add_argument(
        "--force-notebooks",
        action="store_true",
        help="Re-execute per-resolution dataset.ipynb even if a valid output already exists",
    )
    export_parser.add_argument(
        "--force-assets",
        action="store_true",
        help="Re-copy assets/ even if banner.png is present",
    )
    export_parser.add_argument(
        "--force-calibration",
        action="store_true",
        help="Re-copy calibration/ even if calibrate.json is present",
    )
    export_parser.add_argument(
        "--force-analysis-notebooks",
        action="store_true",
        help="Re-copy notebooks/*.ipynb even if already present",
    )
    export_parser.add_argument(
        "--force-metadata",
        action="store_true",
        help="Re-copy LICENSE, croissant.json, DATASHEET.md, CITATION.cff, ... even if present",
    )
    export_parser.add_argument(
        "--force-readme",
        action="store_true",
        help="Re-write README.md even if its content is unchanged",
    )
    export_parser.add_argument(
        "--force-gitattributes",
        action="store_true",
        help="Re-write .gitattributes even if its content matches",
    )
    export_parser.add_argument(
        "--force-strip-outputs",
        action="store_true",
        help="Re-run notebook output stripping (no-op if already stripped, but re-scans every notebook)",
    )
    export_parser.add_argument(
        "--no-strip-outputs",
        action="store_true",
        help="Skip stripping outputs from copied analysis notebooks",
    )
    export_parser.add_argument(
        "--no-gitattributes",
        action="store_true",
        help="Skip writing .gitattributes (HF LFS rules)",
    )
    export_parser.add_argument(
        "--no-validate-croissant",
        action="store_true",
        help="Skip croissant.json validation",
    )
    export_parser.add_argument(
        "--no-validate-readme",
        action="store_true",
        help="Skip README YAML (HF DatasetCard) validation",
    )
    export_parser.add_argument(
        "--no-audit-compression",
        action="store_true",
        help="Skip parquet ZSTD compression audit",
    )
    export_parser.add_argument(
        "--sample-size-warn-mb",
        type=int,
        default=512,
        help="Warn when a *_samples config exceeds this size in MB (default: 512)",
    )
    export_parser.add_argument(
        "--parquet-size-warn-mb",
        type=int,
        default=5000,
        help="Warn when an individual parquet file exceeds this size in MB (default: 5000; HF hard cap is 50000)",
    )
    export_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for multiprocessing (can be specified before or after subcommand)",
    )
    export_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload the prepared export directory to Hugging Face",
    )
    upload_parser.add_argument(
        "--calibrate-dirpath",
        default="outputs/calibrate",
        help="Directory used to derive the export directory (default: outputs/calibrate)",
    )
    upload_parser.add_argument(
        "--calibration-filepath",
        default=None,
        help="Calibration JSON used to derive the export directory (takes precedence over --calibrate-dirpath)",
    )
    upload_parser.add_argument(
        "--export-dirpath",
        default=None,
        help="Explicit export directory override (default: derive from calibration path)",
    )
    upload_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Specific resolutions to upload. If not specified, upload the full prepared export tree.",
    )
    upload_parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id (default: thalesgroup/passage)",
    )
    upload_parser.add_argument(
        "--env-path",
        default=None,
        help="Path to .env file to load before reading HF_TOKEN (default: project root .env)",
    )
    upload_parser.add_argument(
        "--secret-path",
        default=None,
        help="Path to .env.secret file containing HF_TOKEN (default: project root .env.secret)",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded repository private (default: public)",
    )
    upload_parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Don't create the repository if it doesn't exist",
    )
    upload_parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to use for upload workers (can be specified before or after subcommand)",
    )
    upload_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    audit_parser = subparsers.add_parser("audit", help="Audit a released dataset without modifying its files")
    audit_parser.add_argument("--dataset-path", default="dataset", help="Local Hugging Face snapshot root")
    audit_parser.add_argument("--dataset-repo", default="thalesgroup/passage")
    audit_parser.add_argument("--dataset-revision", default=None)
    audit_parser.add_argument("--remote", action="store_true", help="Use the explicit pinned Hugging Face repository")
    audit_parser.add_argument("--resolution", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    audit_parser.add_argument("--split", choices=["train", "calibration", "validation", "test"], default="validation")
    audit_parser.add_argument(
        "--max-samples", type=_positive_int, default=16, help="Bounded core-row audit per resolution"
    )
    audit_parser.add_argument("--max-expansions", type=_positive_int, default=10000)
    audit_parser.add_argument("--output", default="outputs/audit/audit.json")

    qualification_parser = subparsers.add_parser(
        "qualify", help="Authenticate a pinned snapshot, census metadata and audit frozen routes"
    )
    qualification_parser.add_argument("--dataset-path", default="dataset", help="Read-only local snapshot")
    qualification_parser.add_argument("--dataset-repo", default="thalesgroup/passage")
    qualification_parser.add_argument("--dataset-revision", required=True, help="Immutable source commit SHA")
    qualification_parser.add_argument("--output", default="outputs/qualification")
    qualification_parser.add_argument(
        "--phase",
        choices=["all", "authenticate", "census", "routes"],
        default="all",
        help="Run the complete campaign or one resumable phase",
    )
    qualification_parser.add_argument(
        "--workers", type=_positive_int, default=4, help="Bounded hashing/audit workers (1–8)"
    )
    qualification_parser.add_argument(
        "--batch-size", type=_positive_int, default=2048, help="Projected metadata batch size"
    )

    analysis_parser = subparsers.add_parser(
        "analyze", help="Prepare or execute complete release notebooks in separate staging"
    )
    analysis_parser.add_argument("--dataset-path", default="dataset", help="Read-only local snapshot input")
    analysis_parser.add_argument(
        "--staging-root", required=True, help="Separate destination; never use the source snapshot"
    )
    analysis_parser.add_argument("--dataset-repo", default="thalesgroup/passage")
    analysis_parser.add_argument("--dataset-revision", default=None)
    analysis_parser.add_argument(
        "--remote", action="store_true", help="Use pinned remote data rather than the local dataset path"
    )
    analysis_parser.add_argument("--resolution", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    analysis_parser.add_argument("--notebooks-dirpath", default="notebooks")
    analysis_parser.add_argument("--result-root", default=None)
    analysis_parser.add_argument(
        "--qualification-root",
        default=None,
        help="Bound qualification receipts to reuse a complete census and publish its evidence",
    )
    analysis_parser.add_argument(
        "--census-limit",
        type=_positive_int,
        default=None,
        help="Optional smoke-test rows per split; omitted means a full metadata census",
    )
    analysis_parser.add_argument("--max-samples", type=_positive_int, default=1, help="Scene previews per resolution")
    analysis_parser.add_argument(
        "--classical-budget",
        type=_nonnegative_int,
        default=16,
        help="Core validation rows per bounded classical diagnostic; 0 uses no extra solves",
    )
    analysis_parser.add_argument("--max-expansions", type=_positive_int, default=10000)
    analysis_parser.add_argument(
        "--execute", action="store_true", help="Execute sequentially; default prepares files and manifests"
    )
    analysis_parser.add_argument("--cell-timeout", type=_positive_int, default=300)
    analysis_parser.add_argument("--job-timeout", type=_positive_int, default=1800)
    analysis_parser.add_argument("--force", action="store_true", help="Replace stale executed notebook outputs")

    regenerate_parser = subparsers.add_parser(
        "regenerate", help="Generate, export and verify the dataset in bounded, resumable batches; never uploads"
    )
    regenerate_parser.add_argument("--config", default="config/config.yaml")
    regenerate_parser.add_argument("--dataset-dirpath", default="outputs/release/dataset")
    regenerate_parser.add_argument("--scratch-dirpath", default="outputs/release/scratch")
    regenerate_parser.add_argument("--tiles-dirpath", default="outputs/download/tiles")
    regenerate_parser.add_argument("--calibration-filepath", default="outputs/calibrate/calibrate.json")
    regenerate_parser.add_argument("--notebooks-dirpath", default="notebooks")
    regenerate_parser.add_argument("--resolution", type=_positive_int, nargs="+", default=None)
    regenerate_parser.add_argument(
        "--num-cpus",
        type=_nonnegative_int,
        default=None,
        help="Worker ceiling; default floor(75%% of process CPU affinity), 0 serial",
    )
    regenerate_parser.add_argument(
        "--pilot",
        type=_positive_int,
        default=None,
        metavar="SCENES",
        help="Separately identified pilot scenes per resolution; full count/split seeds preserved",
    )
    regenerate_parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs, counts and storage without writing files"
    )
    regenerate_parser.add_argument("--max-raw-bytes", type=_positive_int, default=16 * 1024**3)
    regenerate_parser.add_argument("--min-free-bytes", type=_nonnegative_int, default=64 * 1024**3)
    regenerate_parser.add_argument(
        "--batch-scenes",
        type=_positive_int,
        default=None,
        help="Optional generation stage ceiling; export shard sizes remain configured separately",
    )
    regenerate_parser.add_argument(
        "--export-memory-bytes",
        type=_positive_int,
        default=8 * 1024**3,
        help="Total admitted parallel export workspace; default 8 GiB",
    )

    args = parser.parse_args(argv)

    level = getattr(args, "log_level", "info")
    logger.remove()
    logger.add(sys.stderr, level=level.upper())

    if args.command == "regenerate":
        if run_regenerate is None:
            from passage.regenerate import run_regenerate as _rr

            globals()["run_regenerate"] = _rr
        result = run_regenerate(
            config_filepath=args.config,
            dataset_dirpath=args.dataset_dirpath,
            scratch_dirpath=args.scratch_dirpath,
            tiles_dirpath=args.tiles_dirpath,
            calibration_filepath=args.calibration_filepath,
            notebooks_dirpath=args.notebooks_dirpath,
            resolution=args.resolution,
            num_cpus=args.num_cpus,
            pilot=args.pilot,
            dry_run=args.dry_run,
            max_raw_bytes=args.max_raw_bytes,
            min_free_bytes=args.min_free_bytes,
            batch_scenes=args.batch_scenes,
            export_memory_bytes=args.export_memory_bytes,
        )
        if args.dry_run:
            import json

            print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "audit":
        import json
        from pathlib import Path
        from passage.audit import audit_source
        from passage.dataset import DatasetSource

        source = DatasetSource(
            root=None if args.remote else args.dataset_path, repo=args.dataset_repo, revision=args.dataset_revision
        )
        audits = [
            audit_source(
                source,
                resolution=resolution,
                split=args.split,
                limit=args.max_samples,
                max_expansions=args.max_expansions,
            )
            for resolution in args.resolution
        ]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps({"audits": audits}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        logger.info(f"Audit receipts written to {output}")
    elif args.command == "qualify":
        import json
        from pathlib import Path
        from passage.authentication import authenticate_snapshot
        from passage.dataset import DatasetSource
        from passage.qualification import run_metadata_census

        source = DatasetSource(root=args.dataset_path, repo=args.dataset_repo, revision=args.dataset_revision)
        output = Path(args.output).resolve()
        root = source.root.resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Qualification outputs must be separate from the source snapshot")
        if not 1 <= args.workers <= 8:
            raise ValueError("Qualification workers must be between 1 and 8")
        source_auth = output / "source-authentication"
        if args.phase in {"all", "authenticate"}:
            authentication = authenticate_snapshot(
                root, args.dataset_repo, args.dataset_revision, source_auth, workers=args.workers
            )
            if not authentication["content_verified"]:
                raise RuntimeError("Source authentication remains incomplete; inspect its receipt")
        if args.phase in {"all", "census"}:
            result = run_metadata_census(
                source,
                output / "metadata",
                batch_size=args.batch_size,
                authentication_manifest=source_auth / "manifest.json",
                progress=lambda event: print(json.dumps(event), flush=True),
            )
            if not result["population_scope_complete"]:
                raise RuntimeError("Full census is incomplete; inspect its summary")
        if args.phase in {"all", "routes"}:
            from passage.qualification_runner import run_route_audit

            run_route_audit(source, output / "metadata", output / "route-audit", workers=args.workers)
        logger.info(f"Qualification phase {args.phase} completed; receipts: {output}")
    elif args.command == "analyze":
        from passage.notebook_export import prepare_snapshot_notebooks

        manifest = prepare_snapshot_notebooks(
            None if args.remote else args.dataset_path,
            args.staging_root,
            notebooks_dirpath=args.notebooks_dirpath,
            resolutions=args.resolution,
            repo_id=args.dataset_repo,
            revision=args.dataset_revision,
            execute=args.execute,
            cell_timeout=args.cell_timeout,
            job_timeout=args.job_timeout,
            result_root=args.result_root,
            qualification_root=args.qualification_root,
            force=args.force,
            parameters={
                "census_limit": args.census_limit,
                "max_samples": args.max_samples,
                "classical_budget": args.classical_budget,
                "max_expansions": args.max_expansions,
            },
        )
        logger.info(f"Notebook preparation completed: {manifest.get('status', 'see staging manifest')}")
    elif args.command == "download":
        if run_download is None:
            from passage.download import run_download as _rd

            globals()["run_download"] = _rd
        download_options = {"resume": not args.no_resume, "num_cpus": args.num_cpus}
        if args.archive:
            download_options["archive_pairs"] = [tuple(pair) for pair in args.archive]
        run_download(args.outputs_dirpath, **download_options)
    elif args.command == "generate":
        if run_generate is None:
            from passage.generate import run_generate as _rg

            globals()["run_generate"] = _rg
        run_generate(
            args.outputs_dirpath,
            args.config,
            args.tiles_dirpath,
            calibration_filepath=args.calibration_filepath,
            resolution=args.resolution,
            num_samples=args.num_samples,
            num_cpus=args.num_cpus,
            disable_images=args.no_images,
            force=args.force,
        )
    elif args.command == "calibrate":
        if run_calibrate is None:
            from passage.calibrate import run_calibrate as _rc

            globals()["run_calibrate"] = _rc
        run_calibrate(
            args.outputs_dirpath,
            args.tiles_dirpath,
            resume=not args.no_resume,
            overwrite=args.no_resume,
            num_cpus=args.num_cpus,
        )
    elif args.command == "compress":
        if compress_release is None:
            from passage.export import compress_release as _cr

            globals()["compress_release"] = _cr
        compress_release(
            config_filepath=args.config,
            dataset_dirpath=args.dataset_dirpath,
            calibrate_dirpath=args.calibrate_dirpath,
            calibration_filepath=args.calibration_filepath,
            export_dirpath=args.export_dirpath,
            resolutions=args.resolution,
            num_cpus=args.num_cpus,
        )
    elif args.command == "export":
        if prepare_export is None:
            from passage.export import prepare_export as _pe

            globals()["prepare_export"] = _pe
        from passage.export import ExportSteps

        steps = ExportSteps(
            force_all=args.force,
            force_notebooks=args.force_notebooks,
            force_assets=args.force_assets,
            force_calibration=args.force_calibration,
            force_analysis_notebooks=args.force_analysis_notebooks,
            force_metadata=args.force_metadata,
            force_readme=args.force_readme,
            force_gitattributes=args.force_gitattributes,
            force_strip_outputs=args.force_strip_outputs,
            skip_strip_outputs=args.no_strip_outputs,
            skip_gitattributes=args.no_gitattributes,
            skip_validate_croissant=args.no_validate_croissant,
            skip_validate_readme=args.no_validate_readme,
            skip_audit_compression=args.no_audit_compression,
            sample_size_warn_mb=args.sample_size_warn_mb,
            parquet_size_warn_mb=args.parquet_size_warn_mb,
        )
        prepare_export(
            config_filepath=args.config,
            dataset_dirpath=args.dataset_dirpath,
            calibrate_dirpath=args.calibrate_dirpath,
            calibration_filepath=args.calibration_filepath,
            notebooks_dirpath=args.notebooks_dirpath,
            export_dirpath=args.export_dirpath,
            resolutions=args.resolution,
            repo_id=args.repo_id,
            max_notebook_retries=args.max_notebook_retries,
            notebook_retry_delay=args.notebook_retry_delay,
            num_cpus=args.num_cpus,
            steps=steps,
            keep_existing_configs=args.keep_existing_configs,
        )
    elif args.command == "upload":
        env_display = args.env_path if args.env_path else ".env"
        secret_display = args.secret_path if args.secret_path else ".env.secret"
        logger.info(
            f"Using Hugging Face token from {env_display} and {secret_display}. "
            f"Upload target: {args.repo_id or 'thalesgroup/passage'}."
        )
        if upload_export is None:
            from passage.export import upload_export as _ue

            globals()["upload_export"] = _ue
        upload_export(
            calibrate_dirpath=args.calibrate_dirpath,
            calibration_filepath=args.calibration_filepath,
            export_dirpath=args.export_dirpath,
            resolutions=args.resolution,
            repo_id=args.repo_id,
            env_path=args.env_path,
            secret_path=args.secret_path,
            private=args.private,
            create_repo=not args.no_create_repo,
            num_cpus=args.num_cpus,
        )
    elif args.command is None:
        logger.info("Hello from passage! Use --help for available commands.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
