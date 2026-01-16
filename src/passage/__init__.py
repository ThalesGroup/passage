"""PASSAGE: Path Solving Dataset and Generator."""

import argparse

from loguru import logger

from passage.calibrate import run_calibrate
from passage.download import run_download
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

    args = parser.parse_args()

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
    elif args.command is None:
        logger.info("Hello from passage! Use --help for available commands.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
