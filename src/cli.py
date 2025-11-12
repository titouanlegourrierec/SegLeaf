"""Command-line interface for leaf image processing tools."""

import argparse
import logging
import sys
from pathlib import Path

from .image_processing.batch_processor import split_leaves
from .image_processing.leaf_segmenter import segment_leaves


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_path(path: str, *, should_exist: bool = True) -> str:
    """
    Validate if a path exists or can be created.

    Args:
        path (str): Path to validate
        should_exist (bool): Whether the path should already exist

    Returns:
        str: The validated path

    """
    if should_exist and not Path(path).exists():
        msg = f"Path does not exist: {path}"
        raise argparse.ArgumentTypeError(msg)

    if not should_exist:
        # Check if parent directory exists for output paths
        parent_dir = Path(path).parent
        if parent_dir and not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                msg = f"Cannot create directory: {parent_dir}. Error: {e}"
                raise argparse.ArgumentTypeError(msg) from None
    return path


def validate_color_space(value: str) -> str:
    """
    Validate color space value.

    Args:
        value (str): Color space value to validate

    Returns:
        str: The validated color space value

    """
    valid_spaces = ["RGB", "YUV", "HSV", "LAB", "HLS"]
    value = value.upper()
    if value not in valid_spaces:
        msg = f"Invalid color space: {value}. Valid options are: {', '.join(valid_spaces)}"
        raise argparse.ArgumentTypeError(msg)
    return value


def main() -> None:
    """Parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Leaf image processing tools: segment or split leaves from images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command to split leaves
    parser_split = subparsers.add_parser("split", help="Split leaves in input images into individual leaf images.")
    parser_split.add_argument(
        "input",
        type=lambda x: validate_path(x, should_exist=True),
        help="Path to the input directory containing images to split",
    )
    parser_split.add_argument(
        "output",
        type=lambda x: validate_path(x, should_exist=False),
        help="Path to the output directory where split images will be saved",
    )
    parser_split.add_argument(
        "--color_space",
        "-c",
        type=validate_color_space,
        default="RGB",
        help="Color space to use for saving images. Options: RGB, YUV, HSV, LAB, HLS. Default: RGB",
    )

    # Command to segment leaves
    parser_segment = subparsers.add_parser("segment", help="Segment leaves in input images using a trained model.")
    parser_segment.add_argument(
        "model",
        type=lambda x: validate_path(x, should_exist=True),
        help="Path to the trained model file (.ilp)",
    )
    parser_segment.add_argument(
        "input",
        type=lambda x: validate_path(x, should_exist=True),
        help="Path to the input directory containing images to segment",
    )
    parser_segment.add_argument(
        "output",
        type=lambda x: validate_path(x, should_exist=False),
        help="Path to the output directory where segmented images will be saved",
    )

    args = parser.parse_args()

    try:
        if args.command == "segment":
            segment_leaves(args.model, args.input, args.output)
        elif args.command == "split":
            split_leaves(args.input, args.output, args.color_space)
        else:
            parser.print_help()
    except Exception as e:
        msg = f"Error: {e}"
        logger.exception(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
