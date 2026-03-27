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
export_and_upload = None


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

    # Export subcommand (direct push to Hugging Face datasets)
    export_parser = subparsers.add_parser(
        "export",
        help="Push prepared datasets directly to Hugging Face (zstd shards, no parquet)",
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
        help="Specific resolutions to prepare. If not specified, prepares all found.",
    )
    export_parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id for README and upload commands (default: read from .env)",
    )
    export_parser.add_argument(
        "--no-exec-notebooks",
        action="store_true",
        help="Disable executing dataset.ipynb with papermill",
    )
    export_parser.add_argument(
        "--max-notebook-retries",
        type=int,
        default=3,
        help="Number of attempts to execute dataset.ipynb before giving up; set to 0 for unlimited retries (default: 3)",
    )
    export_parser.add_argument(
        "--notebook-retry-delay",
        type=int,
        default=5,
        help="Seconds to wait between notebook execution retries (default: 5)",
    )
    export_parser.add_argument(
        "--env-path",
        default=None,
        help="Path to .env file containing HF_USER_NAME and HF_REPO_NAME (default: project root .env)",
    )
    export_parser.add_argument(
        "--secret-path",
        default=None,
        help="Path to .env.secret file containing HF_TOKEN (default: project root .env.secret)",
    )
    export_parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Run notebook prep but skip pushing to Hugging Face (useful for local testing)",
    )
    export_parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded repository private (default: public)",
    )
    export_parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Don't create the repository if it doesn't exist",
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

    args = parser.parse_args()

    # Configure global log level (default 'info') when provided on subcommands
    level = getattr(args, "log_level", "info")
    logger.remove()
    logger.add(sys.stderr, level=level.upper())

    if args.command == "download":
        # Prefer module-level patched function if present, otherwise import lazily
        if run_download is None:
            from passage.download import run_download as _rd

            globals()["run_download"] = _rd
        run_download(args.outputs_dirpath, resume=not args.no_resume, num_cpus=args.num_cpus)
    elif args.command == "generate":
        if run_generate is None:
            from passage.generate import run_generate as _rg

            globals()["run_generate"] = _rg
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
    elif args.command == "export":
        # Inform the user when the CLI will read credentials from .env/.env.secret
        if args.repo_id is None:
            env_display = args.env_path if args.env_path else ".env"
            secret_display = args.secret_path if args.secret_path else ".env.secret"
            logger.info(
                f"Using Hugging Face credentials from {env_display} and {secret_display} "
                "(HF_USER_NAME/HF_REPO_NAME and HF_TOKEN). To override, pass --repo-id or --env-path/--secret-path."
            )

        if export_and_upload is None:
            from passage.export import export_and_upload as _ea

            globals()["export_and_upload"] = _ea
        export_and_upload(
            dataset_dirpath=args.dataset_dirpath,
            calibrate_dirpath=args.calibrate_dirpath,
            notebooks_dirpath=args.notebooks_dirpath,
            resolutions=args.resolution,
            repo_id=args.repo_id,
            execute_notebooks=not args.no_exec_notebooks,
            max_notebook_retries=(None if args.max_notebook_retries == 0 else args.max_notebook_retries),
            notebook_retry_delay=args.notebook_retry_delay,
            env_path=args.env_path,
            secret_path=args.secret_path,
            private=args.private,
            create_repo=not args.no_create_repo,
            skip_upload=args.no_upload,
            num_cpus=args.num_cpus,
        )
    elif args.command is None:
        logger.info("Hello from passage! Use --help for available commands.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
