"""Generate evaluation metrics for segmented images by comparing them to ground truth images."""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import pandas as pd
from rich.console import Console

from .generate_groundtruth import load_color_map


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        f1_score,
        jaccard_score,
        precision_score,
        recall_score,
    )
except ImportError:
    logger.exception("Scikit-learn is not installed. Please install it to use the evaluation metrics.")
    sys.exit(1)


# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import DPI as DEFAULT_DPI  # noqa: E402


COLOR_MAP = load_color_map()
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def format_ground_truth(image: np.ndarray, color_map: dict[str, int] = COLOR_MAP) -> np.ndarray:
    """
    Format a grayscale image according to a given color map.

    This function takes a grayscale image and maps pixel values to the closest values in the provided color map.
    (Creating bins between the sorted color map values and assigning each pixel to the nearest color map value.)

    Args:
        image (np.ndarray): Input grayscale image as a NumPy array.
        color_map (dict, optional): Dictionary mapping class names to grayscale values. Defaults to COLOR_MAP.

    Returns:
        np.ndarray: The formatted image where each pixel value corresponds to the closest color map value.

    """
    image = image.astype(np.uint8)
    values = np.array(sorted(color_map.values()))
    bins = np.concatenate(([0], ((values[:-1] + values[1:]) // 2), [256]))
    indices = np.digitize(image, bins) - 1
    return values[np.clip(indices, 0, len(values) - 1)].astype(np.uint8)


def format_images(image: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert both the prediction and ground truth images to flattened grayscale arrays.

    This function takes two images (prediction and ground truth), converts them from RGB to grayscale,
    and flattens them into 1D arrays for further processing or comparison.

    Args:
        image (np.ndarray): The predicted image in RGB format.
        ground_truth (np.ndarray): The ground truth image in RGB format.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the flattened grayscale prediction and ground truth arrays.

    """
    prediction_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).flatten()
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY).flatten()
    return prediction_gray, gt_gray


def calculate_class_metrics(
    gt: np.ndarray, pred: np.ndarray, label: int, dpi: float | None = None
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compute evaluation metrics and area statistics for a given class label.

    This function calculates precision, recall, F1-score, IoU, and area-related metrics for a specific class label
    by comparing ground truth and predicted arrays. The area is computed in mm² using the DPI value.

    Args:
        gt (np.ndarray): Ground truth labels (flattened array).
        pred (np.ndarray): Predicted labels (flattened array).
        label (int): The class label to evaluate.
        dpi (float, optional): Dots per inch value for area calculation. Defaults to DEFAULT_DPI.

    Returns:
        tuple: (
            precision (float): Precision score for the class.
            recall (float): Recall score for the class.
            f1 (float): F1-score for the class.
            iou (float): Intersection over Union for the class.
            gt_area (float): Ground truth area in mm² for the class.
            pred_area (float): Predicted area in mm² for the class.
            surface_error (float): Absolute area difference in mm².
            surface_error_pct (float): Area error as a percentage of ground truth area.
        )

    """
    precision = precision_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    recall = recall_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    f1 = f1_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    iou = jaccard_score(gt, pred, labels=[label], average=None, zero_division=1)[0]

    gt_count = np.sum(gt == label)
    pred_count = np.sum(pred == label)

    # Calculate the area of a pixel in mm² using DPI
    # 1 inch = 25.4 mm, so pixel size in mm = 25.4 / DPI
    # Area of one pixel in mm² = (25.4 / DPI) ** 2
    dpi = dpi if dpi is not None else DEFAULT_DPI
    pixel_area_mm2 = (25.4 / dpi) ** 2
    gt_area = gt_count * pixel_area_mm2
    pred_area = pred_count * pixel_area_mm2
    surface_error = abs(gt_area - pred_area)
    surface_error_pct = (abs(gt_count - pred_count) / gt_count) * 100 if gt_count > 0 else 0

    return (
        precision,
        recall,
        f1,
        iou,
        gt_area,
        pred_area,
        surface_error,
        surface_error_pct,
    )


def process_image_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray, dpi: float | None = None
) -> list[list[float]]:
    """
    Process a single image pair (prediction and ground truth) to compute metrics for each class.

    Args:
        prediction (np.ndarray): The predicted segmented image.
        ground_truth (np.ndarray): The ground truth segmented image.
        dpi (float, optional): Dots per inch value for area calculation. Defaults to None.

    Returns:
        list[list[float]]: A list of metrics for each class.

    """
    ground_truth = format_ground_truth(ground_truth)
    pred, gt = format_images(prediction, ground_truth)

    all_metrics = []
    for class_name, label in COLOR_MAP.items():
        metrics = calculate_class_metrics(gt, pred, label, dpi=dpi)
        all_metrics.append([class_name, *list(metrics)])
    return all_metrics


def process_single_file(
    entry: str,
    segmented_images_dir: str = "./evaluation/segmented_images",
    groundtruth_dir: str = "./evaluation/ground_truth",
    dpi: float | None = None,
    console: Console = None,
) -> list[dict[str, float]]:
    """
    Process a single image file to compute evaluation metrics.

    Args:
        entry (str): The file entry to process.
        segmented_images_dir (str, optional): Directory containing segmented images. Defaults to
        './evaluation/segmented_images'.
        groundtruth_dir (str, optional): Directory containing ground truth images. Defaults to
        './evaluation/ground_truth'.
        dpi (float, optional): Dots per inch value for area calculation. Defaults to None.
        console (Console, optional): Rich console for logging. Defaults to None.

    Returns:
        list[dict[str, float]]: A list of dictionaries containing metrics for each class.

    """
    file_name, ext = os.path.splitext(entry.name)
    if ext.lower() not in VALID_EXTENSIONS:
        return []

    prediction_path = os.path.join(segmented_images_dir, f"{file_name}.png")
    ground_truth_path = os.path.join(groundtruth_dir, f"{file_name}{ext}")

    if console:
        console.print(f"[cyan]Processing image:[/] [bold]{file_name}[/]")
    else:
        msg = f"Processing image: {file_name}"
        logger.info(msg)

    if not os.path.exists(prediction_path) or not os.path.exists(ground_truth_path):
        error_msg = f"Missing prediction or ground truth for {file_name}"
        if console:
            console.print(f"[bold red]{error_msg}[/]")
        else:
            logger.error(error_msg)
        return []

    try:
        prediction_img = cv2.cvtColor(cv2.imread(prediction_path), cv2.COLOR_BGR2RGB)
        ground_truth = cv2.cvtColor(cv2.imread(ground_truth_path), cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        error_msg = f"Error reading images for {file_name}: {e}"
        if console:
            console.print(f"[bold red]{error_msg}[/]")
        else:
            logger.exception(error_msg)
        return []

    if console:
        console.print(f"  [green]Calculating metrics for[/] [bold]{file_name}[/]")
    else:
        msg = f"  Calculating metrics for {file_name}"
        logger.info(msg)

    metrics = process_image_metrics(prediction_img, ground_truth, dpi=dpi)
    rows = []

    for class_metrics in metrics:
        class_name = class_metrics[0]
        metric_values = class_metrics[1:]
        row = {"image": file_name, "class": class_name}
        for key, val in zip(
            [
                "precision",
                "recall",
                "f1",
                "IoU",
                "ground_truth_surface_mm2",
                "prediction_surface_mm2",
                "surface_error_mm2",
                "surface_error_percent",
            ],
            metric_values,
            strict=False,
        ):
            # Round all numeric values to 3 decimal places
            row[key] = round(val, 3) if isinstance(val, (int, float)) else val
        rows.append(row)

    if console:
        console.print(f"  [bold green]Completed processing for[/] [bold]{file_name}[/]")
    else:
        msg = f"  Completed processing for {file_name}"
        logger.info(msg)

    return rows


def collect_image_metrics(
    images_to_segment_dir: str = "./evaluation/images_to_segment",
    segmented_images_dir: str = "./evaluation/segmented_images",
    groundtruth_dir: str = "./evaluation/ground_truth",
    dpi: float | None = None,
) -> pd.DataFrame:
    """
    Collect evaluation metrics for all images in a directory using parallel processing.

    This function scans the directory containing images to segment, processes each image by comparing the predicted
    segmentation with the ground truth, and aggregates the results into a pandas DataFrame. The processing is
    parallelized for efficiency.

    Args:
        images_to_segment_dir (str, optional): Path to the directory containing images to be segmented. Defaults to
        './evaluation/images_to_segment'.
        segmented_images_dir (str, optional): Path to the directory containing segmented (predicted) images. Defaults
        to './evaluation/segmented_images'.
        groundtruth_dir (str, optional): Path to the directory containing ground truth images. Defaults to
        './evaluation/ground_truth'.
        dpi (float | None, optional): DPI (dots per inch) value to use for area calculations. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each image and class.

    """
    console = Console()

    console.print(f"[bold]Scanning directory:[/] {images_to_segment_dir}")
    entries = [entry for entry in os.scandir(images_to_segment_dir) if entry.is_file()]
    console.print(f"[bold green]Found {len(entries)} files to process[/]")
    all_data = []

    with console.status("[bold green]Calculating metrics..."), ThreadPoolExecutor(max_workers=8) as executor:
        process_func = partial(
            process_single_file,
            segmented_images_dir=segmented_images_dir,
            groundtruth_dir=groundtruth_dir,
            dpi=dpi,
            console=console,
        )
        results = executor.map(process_func, entries)

        for res in results:
            all_data.extend(res)

    console.print(
        f"[bold green]Processing complete! [/]Generated metrics for {len(all_data)} class-image combinations"
    )

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation metrics for segmented images.")

    parser.add_argument(
        "--images_to_segment_dir",
        type=str,
        default="./evaluation/images_to_segment",
        help="Path to the directory containing images to be segmented.",
    )

    parser.add_argument(
        "--segmented_images_dir",
        type=str,
        default="./evaluation/segmented_images",
        help="Path to the directory containing segmented (predicted) images.",
    )

    parser.add_argument(
        "--groundtruth_dir",
        type=str,
        default="./evaluation/ground_truth",
        help="Path to the directory containing ground truth images.",
    )

    parser.add_argument(
        "--output_csv_path",
        type=str,
        default="./evaluation/metrics_output.csv",
        help="Path to save the output CSV file containing evaluation metrics.",
    )

    parser.add_argument(
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help=f"DPI (dots per inch) value to use for area calculations. Default: {DEFAULT_DPI}",
    )

    args = parser.parse_args()

    console = Console()
    console.print("[bold green]Starting evaluation metrics generation...[/]")
    console.print(f"[bold]Images to segment directory:[/] {args.images_to_segment_dir}")
    console.print(f"[bold]Segmented images directory:[/] {args.segmented_images_dir}")
    console.print(f"[bold]Ground truth directory:[/] {args.groundtruth_dir}")
    console.print(f"[bold]DPI value:[/] {args.dpi}")

    df = collect_image_metrics(
        images_to_segment_dir=args.images_to_segment_dir,
        segmented_images_dir=args.segmented_images_dir,
        groundtruth_dir=args.groundtruth_dir,
        dpi=args.dpi,
    )

    console.print(f"[bold]Writing results to CSV:[/] {args.output_csv_path}")
    df.to_csv(args.output_csv_path, index=False)
    console.print(f"[bold green]Metrics saved to {args.output_csv_path}[/]")
    console.print("[bold green]Evaluation complete![/]")
