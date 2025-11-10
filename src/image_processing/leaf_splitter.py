"""Module for splitting leaves from images and saving the cropped parts."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from src.config import (
    AREA_RATIO_THRESHOLD,
    BINARY_THRESHOLD,
    BLUR_KERNEL,
    MARGIN,
    MAX_BINARY_VALUE,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeafSplitter:
    """
    A class to detect and split leaves (or similar objects) from an image and save the cropped parts.

    Attributes:
        img_path (Path): Path to the input image.
        output_dir (Path): Directory where the cropped images will be saved.
        color_space (str): Color space to save the output images ('RGB', 'YUV', 'HSV', 'LAB', 'HLS').
        blur_kernel (Tuple[int, int]): Kernel size for blurring the image.
        binary_threshold (int): Threshold value for binarization.
        max_binary_value (int): Maximum value for binary thresholding.
        area_ratio_threshold (float): Minimum area ratio to consider a contour as a valid object.
        margin (int): Margin to add around detected bounding boxes.

    """

    def __init__(
        self,
        img_path: str,
        output_dir: str,
        color_space: str = "RGB",
        blur_kernel: tuple[int, int] = BLUR_KERNEL,
        binary_threshold: int = BINARY_THRESHOLD,
        max_binary_value: int = MAX_BINARY_VALUE,
        area_ratio_threshold: float = AREA_RATIO_THRESHOLD,
        margin: int = MARGIN,
    ) -> None:
        """Initialize the LeafSplitter."""
        self.img_path: Path = Path(img_path)
        self.output_dir: Path = Path(output_dir)
        self.color_space: str = color_space.upper()
        self.blur_kernel: tuple[int, int] = blur_kernel
        self.binary_threshold: int = binary_threshold
        self.max_binary_value: int = max_binary_value
        self.area_ratio_threshold: float = area_ratio_threshold
        self.margin: int = margin
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_image(self) -> np.ndarray | None:
        """
        Read the image using PIL and convert it to OpenCV format.

        Returns:
            np.ndarray | None: Image as a NumPy array in BGR format for OpenCV or None if reading fails

        """
        # Disable DecompressionBomb safety check for large images
        PILImage.MAX_IMAGE_PIXELS = None

        # Use PIL to read the image
        try:
            pil_img = PILImage.open(str(self.img_path))
            img = np.array(pil_img.convert("RGB"))
            # Convert from RGB to BGR for OpenCV compatibility
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except (FileNotFoundError, OSError) as e:
            msg = f"Error reading image {self.img_path}: {e!s}"
            logger.exception(msg)
            return None

    def detect_leaf_boxes(self, img: np.ndarray) -> list[list[int]]:
        """
        Detect bounding boxes of leaves (or objects) in the image.

        Args:
            img (np.ndarray): Input image (as a NumPy array, BGR format).

        Returns:
            list[list[int]]: List of bounding boxes, each as [x1, y1, x2, y2].

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, self.blur_kernel)
        _, binary = cv2.threshold(blurred, self.binary_threshold, self.max_binary_value, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return []
        max_area = max(areas)
        bounding_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.area_ratio_threshold * max_area:
                x, y, w, h = cv2.boundingRect(c)
                x1 = max(x - self.margin, 0)
                y1 = max(y - self.margin, 0)
                x2 = min(x + w + self.margin, img.shape[1])
                y2 = min(y + h + self.margin, img.shape[0])
                bounding_boxes.append([x1, y1, x2, y2])
        return bounding_boxes

    def split_and_save(self) -> None:
        """
        Split the image into detected parts and save each part as a separate file in the output directory.

        Returns:
            None

        """
        # Use our helper method to read the image
        img = self.read_image()

        if img is None:
            msg = f"Image not found or could not be read: {self.img_path}"
            raise FileNotFoundError(msg)
        bounding_boxes = self.detect_leaf_boxes(img)
        for i, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box
            part = img[y1:y2, x1:x2]
            # Ajouter le nom de l'espace couleur au nom du fichier
            output_path = (
                self.output_dir / f"{self.img_path.stem}_{i + 1}_{self.color_space.upper()}{self.img_path.suffix}"
            )
            if self.color_space == "RGB":
                # Convert BGR to RGB and save with PIL
                part_rgb = cv2.cvtColor(part, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(part_rgb)
                pil_img.save(str(output_path))
            else:
                part_converted = self._convert_color_space(part)
                cv2.imwrite(str(output_path), part_converted)

    def _convert_color_space(self, img: np.ndarray) -> np.ndarray:
        """Convert the image to the desired color space for saving."""
        cs = self.color_space
        if cs == "RGB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if cs == "YUV":
            return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if cs == "HSV":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if cs == "LAB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        if cs == "HLS":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # Default: no conversion
        return img
