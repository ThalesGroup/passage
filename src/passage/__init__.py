"""PASSAGE: Path Solving Dataset and Generator."""

import argparse
import sys

from loguru import logger

from passage.calibrate import run_calibrate
from passage.download import run_download
from passage.export import prepare_export, run_export
from passage.generate import run_generate


def main() -> None:
    """Entry point for the PASSAGE CLI."""
    parser = argparse.ArgumentParser(
        prog="passage",
        description="PASSAGE: Path Solving Dataset and Generator",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Download subcommand
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

    # Generate subcommand
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

    # Calibrate subcommand
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

    # Prepare subcommand (local structure + manifests, no upload)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare outputs/Export from outputs/dataset (no upload)",
    )
    prepare_parser.add_argument(
        "--dataset-dirpath",
        default="outputs/dataset",
        help="Directory containing generated datasets (default: outputs/dataset)",
    )
    prepare_parser.add_argument(
        "--export-dirpath",
        default="outputs/Export",
        help="Directory to write prepared export structure (default: outputs/Export)",
    )
    prepare_parser.add_argument(
        "--calibrate-dirpath",
        default="outputs/calibrate",
        help="Directory containing calibration data (default: outputs/calibrate)",
    )
    prepare_parser.add_argument(
        "--notebooks-dirpath",
        default="notebooks",
        help="Directory containing notebooks (default: notebooks)",
    )
    prepare_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Specific resolutions to prepare. If not specified, prepares all found.",
    )
    prepare_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    # Export subcommand (upload only)
    export_parser = subparsers.add_parser(
        "export",
        help="Upload datasets to Hugging Face Hub",
    )
    export_parser.add_argument(
        "repo_prefix",
        help="Hugging Face repository prefix (e.g., 'username/passage').",
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
        "--notebooks-dirpath",
        default="notebooks",
        help="Directory containing notebooks (default: notebooks)",
    )
    export_parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Specific resolutions to export. If not specified, exports all found.",
    )
    export_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip exporting calibration data",
    )
    export_parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Also export notebooks as documentation",
    )
    export_parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip exporting PNG images (reduces upload size)",
    )
    export_parser.add_argument(
        "--no-manifests",
        action="store_true",
        help="Skip regenerating parquet manifests before upload",
    )
    export_parser.add_argument(
        "--private",
        action="store_true",
        help="Make repositories private",
    )
    export_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )

    args = parser.parse_args()

    # Configure global log level (default 'info') when provided on subcommands
    level = getattr(args, "log_level", "info")
    logger.remove()
    logger.add(sys.stderr, level=level.upper())

    if args.command == "download":
        run_download(args.outputs_dirpath, resume=not args.no_resume, num_cpus=args.num_cpus)
    elif args.command == "generate":
        run_generate(
            args.outputs_dirpath,
            args.config,
            args.tiles_dirpath,
            resolution=args.resolution,
            num_samples=args.num_samples,
            num_cpus=args.num_cpus,
            disable_images=args.no_images,
            force=args.force,
        )
    elif args.command == "calibrate":
        # When --no-resume is set, user expects a full recompute: set overwrite=True
        run_calibrate(
            args.outputs_dirpath,
            args.tiles_dirpath,
            resume=not args.no_resume,
            overwrite=args.no_resume,
            num_cpus=args.num_cpus,
        )
    elif args.command == "prepare":
        prepare_export(
            export_dirpath=args.export_dirpath,
            dataset_dirpath=args.dataset_dirpath,
            calibrate_dirpath=args.calibrate_dirpath,
            notebooks_dirpath=args.notebooks_dirpath,
            resolutions=args.resolution,
            repo_prefix="passage",
        )
    elif args.command == "export":
        repo_prefix = args.repo_prefix
        run_export(
            repo_prefix=repo_prefix,
            dataset_dirpath=args.dataset_dirpath,
            calibrate_dirpath=args.calibrate_dirpath,
            notebooks_dirpath=args.notebooks_dirpath,
            resolutions=args.resolution,
            include_calibration=not args.no_calibration,
            include_notebooks=args.include_notebooks,
            include_images=not args.no_images,
            private=args.private,
            write_manifests=not args.no_manifests,
        )
    elif args.command is None:
        logger.info("Hello from passage! Use --help for available commands.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
