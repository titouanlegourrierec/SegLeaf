"""Configuration constants for the SegLeaf project."""

# Scanner resolution (dpi) - default value
DPI: int = 1200

# Image processing constants for leaf splitting
BLUR_KERNEL: tuple[int, int] = (
    80,
    80,
)  # Kernel size for blurring the image before thresholding
BINARY_THRESHOLD: int = 128  # Threshold value for binarization
MAX_BINARY_VALUE: int = 255  # Maximum value for binary thresholding
AREA_RATIO_THRESHOLD: float = 0.1  # Minimum area ratio to consider a contour as a valid object
MARGIN: int = 20  # Margin to add around detected bounding boxes
