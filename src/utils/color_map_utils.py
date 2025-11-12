"""Utilities for working with the color map."""

import json
from pathlib import Path

import numpy as np


def get_color_map() -> dict[str, int]:
    """
    Load the color map from the JSON file.

    Returns:
        dict: A dictionary mapping class names to pixel values.

    """
    color_map_path = Path(__file__).parent.parent / "color_map.json"
    try:
        with color_map_path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default color map if file doesn't exist or is invalid
        return {"bg": 0, "leaf": 128}


def save_color_map(color_map: dict[str, int]) -> None:
    """
    Save the color map to the JSON file.

    Args:
        color_map (dict): A dictionary mapping class names to pixel values.

    """
    color_map_path = Path(__file__).parent.parent / "color_map.json"
    with color_map_path.open("w") as f:
        json.dump(color_map, f, indent=4)


def get_class_from_pixel_value(pixel_value: int, color_map: dict[str, int] | None = None) -> str:
    """
    Get the class name for a given pixel value.

    Args:
        pixel_value (int): The pixel value.
        color_map (dict, optional): A dictionary mapping class names to pixel values.
            If not provided, it will be loaded from the file.

    Returns:
        str: The class name corresponding to the pixel value, or 'unknown' if not found.

    """
    if color_map is None:
        color_map = get_color_map()

    # Create a reverse mapping from values to class names
    reverse_map = {v: k for k, v in color_map.items()}

    # Return the class name or 'unknown'
    return reverse_map.get(pixel_value, "unknown")


def generate_segmentation_results(
    segmented_img: np.ndarray, color_map: dict[str, int] | None = None
) -> dict[str, int]:
    """
    Generate segmentation results with proper class names.

    Args:
        segmented_img: The segmented image as a numpy array.
        color_map (dict, optional): A dictionary mapping class names to pixel values.
            If not provided, it will be loaded from the file.

    Returns:
        dict: A dictionary mapping class names to areas in pixels.

    """
    if color_map is None:
        color_map = get_color_map()

    # Create a reverse mapping from values to class names
    reverse_map = {v: k for k, v in color_map.items()}

    # Count pixels for each value
    unique, counts = np.unique(segmented_img, return_counts=True)

    # Map to class names
    results = {}
    for val, count in zip(unique, counts, strict=False):
        class_name = reverse_map.get(val, f"unknown_{val}")
        results[class_name] = count

    return results
