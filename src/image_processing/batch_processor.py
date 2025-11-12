"""Batch image processing using LeafSplitter."""

import csv
import logging
from pathlib import Path

from PIL import Image
from rich.console import Console

from src import config
from src.image_processing.leaf_splitter import LeafSplitter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_leaves(input_dir: str, output_dir: str, color_space: str = "RGB") -> None:
    """
    Split leaves in all images within the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where split images will be saved.
        color_space (str): Color space to use for saving images ('RGB', 'YUV', 'HSV', 'LAB', 'HLS').

    """
    console = Console()
    processor = BatchImageProcessor(input_dir, output_dir, color_space)
    is_valid, error_message = processor.validate_directories()
    if not is_valid:
        msg = f"Directory validation failed: {error_message}"
        raise ValueError(msg)

    with console.status("[bold green]Processing images..."):
        processor.process_images(console=console)


class BatchImageProcessor:
    """Class for batch processing of images using LeafSplitter."""

    def __init__(self, input_dir: str, output_dir: str, color_space: str = "RGB") -> None:
        """Initialize the BatchImageProcessor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.color_space = color_space.upper()

    def validate_directories(self) -> tuple[bool, str]:
        """
        Validate input and output directories.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)

        """
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            return False, "Input directory does not exist."
        if not self.output_dir.exists() or not self.output_dir.is_dir():
            return False, "Output directory does not exist."
        return True, ""

    def find_images(self) -> list[Path]:
        """
        Find all image files in the input directory.

        Returns:
            List[Path]: List of image file paths

        """
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
        return [img_path for ext in image_extensions for img_path in self.input_dir.glob(ext) if img_path.is_file()]

    def process_images(self, progress_callback: callable | None = None, console: Console | None = None) -> None:
        """
        Process all images in the input directory.

        Args:
            progress_callback: Optional callback function to report progress
            console: Optional Rich console for displaying progress

        """
        images = self.find_images()

        # List to store information about processed images for CSV
        image_data = []

        # Get original image dimensions
        orig_image_dimensions = {}
        for img_path in images:
            try:
                # Désactiver la limite de taille d'image pour les images très grandes
                Image.MAX_IMAGE_PIXELS = None

                with Image.open(img_path) as orig_img:
                    orig_width, orig_height = orig_img.size
                    orig_image_dimensions[img_path.name] = (orig_width, orig_height)
            except (OSError, ValueError) as e:  # noqa: PERF203
                if console:
                    console.log(f"[bold red]Error reading original image {img_path.name}: {e!s}")
                continue

        for idx, img_path in enumerate(images, 1):
            try:
                # Skip if we couldn't get dimensions
                if img_path.name not in orig_image_dimensions:
                    continue

                orig_width, orig_height = orig_image_dimensions[img_path.name]

                # Create the splitter and get the image
                splitter = LeafSplitter(
                    str(img_path),
                    str(self.output_dir),
                    color_space=self.color_space,
                )

                # Get the bounding boxes before processing
                img = splitter.read_image()
                if img is None:
                    if console:
                        console.log(f"[bold red]Error reading image {img_path.name}")
                    continue

                bounding_boxes = splitter.detect_leaf_boxes(img)

                # Process and save the image
                splitter.split_and_save()

                # Collect information for each segmented part
                for i, box in enumerate(bounding_boxes):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1

                    part_name = f"{img_path.stem}_{i + 1}_{self.color_space.upper()}{img_path.suffix}"

                    image_data.append(
                        {
                            "part_image": part_name,
                            "original_image": img_path.name,
                            "part_width": width,
                            "part_height": height,
                            "original_width": orig_width,
                            "original_height": orig_height,
                        }
                    )

                if console:
                    console.log(f"[bold green]Processed image {idx}/{len(images)}: {img_path.name}")
            except (OSError, ValueError) as e:
                if console:
                    console.log(f"[bold red]Error processing {img_path.name}: {e!s}")

            if progress_callback:
                progress_callback(idx, len(images))

        # Save the image data to CSV
        self._save_image_data_csv(image_data)

    def _save_image_data_csv(self, image_data: list[dict]) -> None:
        """
        Save image data to a CSV file in the output directory.

        Args:
            image_data: List of dictionaries containing image data

        """
        if not image_data:
            return

        csv_path = self.output_dir / "parts_data.csv"

        fieldnames = [
            "part_image",
            "original_image",
            "part_width",
            "part_height",
            "original_width",
            "original_height",
        ]

        with csv_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(image_data)

    def get_output_stats(
        self,
    ) -> tuple[list[int] | None, list[int] | None, list[Path]]:
        """
        Compute statistics about the processed images.

        Returns:
            Tuple[Optional[List[int]], Optional[List[int]], List[Path]]:
            (widths, heights, image_paths)

        """
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
        output_images = [f for ext in image_extensions for f in self.output_dir.glob(ext) if f.is_file()]

        widths, heights = [], []
        Image.MAX_IMAGE_PIXELS = None
        for img_path in output_images:
            try:
                with Image.open(img_path) as im:
                    # Convert pixels to millimeters: mm = (pixels / DPI) * 25.4
                    widths.append(im.width / config.DPI * 25.4)
                    heights.append(im.height / config.DPI * 25.4)
            except (OSError, ValueError) as e:  # noqa: PERF203
                msg = f"Error processing image {img_path}: {e!s}"
                logger.exception(msg)
                continue

        return widths, heights, output_images
