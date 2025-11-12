"""Command-line interface for leaf segmentation using Ilastik models."""

import argparse
import logging
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add the parent directory to sys.path to allow importing the src module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.image_processing.leaf_segmenter import segment_leaves
except ImportError:
    logger.exception("Could not import segment_leaves. Make sure the module exists.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description="Segment leaf images using a trained Ilastik model.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained Ilastik model file (.ilp)",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="./evaluation/images_to_segment",
        help="Path to the directory containing input images to segment (default: ./evaluation/images_to_segment)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation/segmented_images",
        help="Path to the directory where segmented images will be saved (default: ./evaluation/segmented_images)",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


def main() -> None:
    """Execute leaf segmentation based on command-line arguments."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate paths
    for path_name, path_value in [
        ("model_path", args.model_path),
        ("input_path", args.input_path),
    ]:
        if not Path(path_value).exists():
            msg = f"Error: The specified {path_name} does not exist: {path_value}"
            logger.error(msg)
            sys.exit(1)

    # Create output directory if it doesn't exist
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Check if model file has the right extension
    if not args.model_path.lower().endswith(".ilp"):
        msg = f"Warning: model_path '{args.model_path}' does not have .ilp extension."
        logger.warning(msg)

    try:
        msg = (
            f"Starting leaf segmentation with model: {args.model_path}\n"
            f"Input directory: {args.input_path}\n"
            f"Output directory: {args.output_path}"
        )

        logger.info(msg)

        # Run segmentation
        segment_leaves(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
        )

        msg = (
            f"Segmentation completed successfully. Results saved to {args.output_path}\n"
            "A CSV report with pixel class statistics has been generated at"
            f"{Path(args.output_path) / 'results.csv'}"
        )
        logger.info(msg)
    except Exception as e:
        msg = f"An error occurred during segmentation: {e!s}"
        logger.exception(msg)
        if args.verbose:
            import traceback  # noqa: PLC0415

            logger.exception(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
