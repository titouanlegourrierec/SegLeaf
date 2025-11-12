"""Module for segmenting leaves in images using a trained Ilastik model and generating pixel class statistics."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from EasIlastik.run_ilastik import run_ilastik
from PIL import Image

from src import config


def segment_leaves(model_path: str, input_path: str, output_path: str) -> None:
    """
    Segment leaves in images using a trained Ilastik model.

    Args:
        model_path (str): Path to the trained model file.
        input_path (str): Path to the input image directory.
        output_path (str): Path to save the segmented output images.

    """
    segmenter = LeafSegmenter(model_path, input_path, output_path)
    segmenter.segmenter(input_path, output_path)
    segmenter.make_bilan()


class LeafSegmenter:
    """
    A class to perform image segmentation using a trained Ilastik model and to generate pixel class statistics.

    Attributes:
        model_path (str): Path to the Ilastik model file.
        input_path (str, optional): Path to the input images directory.
        output_path (str, optional): Path to the output directory for segmented images.

    """

    def __init__(
        self,
        model_path: str,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> None:
        """Initialize the LeafSegmenter."""
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path

    def segmenter(self, input_path: str | None = None, output_path: str | None = None) -> None:
        """
        Run Ilastik segmentation on images in the input directory and save results to the output directory.

        Optionally, input and output paths can be provided; otherwise, instance attributes are used.
        After segmentation, renames files to remove '_Simple_Segmentation' from filenames.
        Temporarily moves any CSV files out of the input directory during Ilastik processing.

        Args:
            input_path (str, optional): Path to the input images directory.
            output_path (str, optional): Path to the output directory for segmented images.

        Raises:
            ValueError: If input_path or output_path is not specified.

        """
        # Use class attributes if arguments are not provided
        if input_path is None:
            input_path = self.input_path
        if output_path is None:
            output_path = self.output_path
        if input_path is None or output_path is None:
            msg = "input_path and output_path must be specified."
            raise ValueError(msg)
        # Ensure output_path ends with a slash
        if not output_path.endswith("/"):
            output_path += "/"
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        # Temporarily move CSV files out of the input directory
        temp_dir = tempfile.mkdtemp()
        csv_files = []

        # for fname in os.listdir(input_path):
        for fname in input_path.iterdir():
            if fname.name.lower().endswith(".csv"):
                src = input_path / fname
                dst = Path(temp_dir) / fname.name
                src.rename(dst)
                csv_files.append((src, dst))

        try:
            # Run Ilastik without CSV files
            run_ilastik(
                input_path=input_path,
                model_path=self.model_path,
                result_base_path=output_path,
            )

            # Rename files to remove '_Simple_Segmentation' from filenames
            for fname in output_path.iterdir():
                if "_Simple_Segmentation" in fname.name:
                    new_name = fname.name.replace("_Simple_Segmentation", "")
                    src = fname
                    dst = output_path / new_name
                    if not dst.exists():
                        src.rename(dst)
        finally:
            # Move CSV files back to input directory
            for src, dst in csv_files:
                if dst.exists():
                    dst.rename(src)

    def make_bilan(self) -> None:
        """
        For each segmented image in output_path, count pixels per class and save the results to a CSV file.

        Each row in the CSV corresponds to an image, and each column to a class (using class names from color_map.json
        if available).

        Raises:
            ValueError: If output_path is not set.

        """
        if self.output_path is None:
            msg = "output_path must be set to count pixels per class."
            raise ValueError(msg)
        segmented_dir = self.output_path
        image_files = [f.name for f in segmented_dir.iterdir() if f.name.lower().endswith((".png", ".tif", ".tiff"))]
        image_files.sort()

        # Collect all unique classes present in all images
        all_classes: set[int] = set()
        image_class_areas = {}
        mm2_per_pixel = (25.4 / config.DPI) ** 2
        for img_file in image_files:
            img_path = segmented_dir / img_file
            img = Image.open(img_path)
            arr = np.array(img)
            unique, counts = np.unique(arr, return_counts=True)
            # Surface en mm² pour chaque classe
            class_area = {int(u): float(c) * mm2_per_pixel for u, c in zip(unique, counts, strict=False)}
            image_class_areas[img_file] = class_area
            all_classes.update(class_area.keys())

        all_classes_sorted: list[int] = sorted(all_classes)

        # Read color_map.json to get class names
        color_map_path = Path(__file__).parent.parent / "color_map.json"
        with color_map_path.open() as f:
            color_map = json.load(f)
        # Invert the mapping to get {value: name}
        value_to_name = {v: k for k, v in color_map.items()}

        # Prepare column order: class names if known, otherwise use raw value
        header_classes = []
        for c in all_classes_sorted:
            name = value_to_name.get(c, str(c))
            header_classes.append(name)

        # Write the CSV file (surface in mm²)
        csv_path = Path(segmented_dir) / "results.csv"
        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["Image", *header_classes]
            writer.writerow(header)
            for img_file in image_files:
                row = [img_file]
                class_area = image_class_areas[img_file]
                for c in all_classes_sorted:
                    value = class_area.get(c, 0.0)
                    row.append(f"{value:.2f}")
                writer.writerow(row)
