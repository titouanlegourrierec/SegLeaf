"""Frame for displaying segmentation statistics."""

import csv
import logging
import os
import tkinter as tk
from tkinter import ttk

import cv2
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationStatsFrame(ttk.Frame):
    """Frame for displaying segmentation statistics."""

    def update_stats(self) -> None:
        """Update the statistics plot."""
        self.load_stats()

    def __init__(
        self,
        parent: tk.Widget,
        output_dir_var: tk.StringVar,
        input_dir_var: tk.StringVar | None = None,
    ) -> None:
        """Initialize the statistics frame."""
        super().__init__(parent)
        self.output_dir_var = output_dir_var
        self.input_dir_var = input_dir_var  # Variable for the input directory
        self.class_var = tk.StringVar()
        self.class_options: list[str] = []
        self.stats_canvas = None
        self._img_popup = None
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        """Set up the widgets in the statistics frame."""
        # Dropdown for class selection
        self.dropdown = ttk.Combobox(self, textvariable=self.class_var, state="readonly")
        self.dropdown.pack(pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", lambda _: self.plot_stats())

    def load_stats(self) -> None:
        """Load statistics from CSV file."""
        csv_path = os.path.join(self.output_dir_var.get(), "results.csv")
        if not os.path.exists(csv_path):
            return
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
            self.class_options = [col for col in reader.fieldnames if col != "Image"]
        self.dropdown["values"] = self.class_options
        if self.class_options:
            self.class_var.set(self.class_options[0])
            self.plot_stats()

    def plot_stats(self) -> None:
        """Plot the statistics."""
        if not hasattr(self, "rows") or not self.class_var.get():
            return
        areas = []
        props = []
        image_paths = []
        output_dir = self.output_dir_var.get()

        for row in self.rows:
            total = sum(float(row[c]) for c in self.class_options)
            area = float(row[self.class_var.get()])
            prop = area / total if total > 0 else 0
            areas.append(area)
            props.append(prop)
            img_name = row["Image"]
            # If it's not an absolute path, take it from the output folder
            img_path = os.path.join(output_dir, img_name) if not os.path.isabs(img_name) else img_name
            image_paths.append(img_path)
        # Remove old plot
        if self.stats_canvas:
            self.stats_canvas.get_tk_widget().pack_forget()

        # Create figure with a grid of subplots: scatter plot in center, histograms on sides
        fig = plt.figure(figsize=(10, 8))

        # Create a grid with ratios (larger center plot, smaller histograms)
        gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), hspace=0.02, wspace=0.02)

        # Create three axes
        ax_scatter = fig.add_subplot(gs[1, 0])  # main scatter plot
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)  # x-axis histogram
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)  # y-axis histogram

        # Hide the labels on the histograms
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # Scatter plot
        scatter = ax_scatter.scatter(areas, props, alpha=0.8, color="blue")
        ax_scatter.set_xlabel("Class Area (mm²)")
        ax_scatter.set_ylabel("Proportion of Total Area")
        ax_scatter.set_xlim(left=0)

        # Adjust the y-axis scale to the maximum proportion with a small margin of 5%
        if props:
            max_prop = max(props)
            ax_scatter.set_ylim(0, max_prop * 1.05)  # 5% margin above the max value
        else:
            ax_scatter.set_ylim(0, 1)  # Default scale if no data

        # Histograms
        bins = min(int(len(areas) / 2), 30)  # More bins for better granularity
        if bins < 10:  # Ensure at least 10 bins if we have enough data points  # noqa: PLR2004
            bins = max(10, len(areas) // 2)

        ax_histx.hist(areas, bins=bins, alpha=0.8, color="blue", edgecolor="black")
        ax_histy.hist(
            props,
            bins=bins,
            alpha=0.8,
            color="blue",
            edgecolor="black",
            orientation="horizontal",
        )

        # Adjust the layout to minimize empty space
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.95)

        cursor = mplcursors.cursor(scatter, hover=False)

        def show_img_on_click(sel: mplcursors.Selection) -> None:
            """Show image preview when clicking a point in the scatter plot."""
            idx = sel.index
            if self._img_popup is not None and self._img_popup.winfo_exists():
                self._img_popup.destroy()
            if 0 <= idx < len(image_paths):
                try:
                    # Load the output image
                    output_img_path = image_paths[idx]
                    output_img = Image.open(output_img_path)
                    output_orig_img = output_img.copy()

                    # Find and load the corresponding input image
                    output_basename = os.path.basename(output_img_path)
                    output_name, _ = os.path.splitext(output_basename)

                    # Get the input directory
                    input_dir = ""
                    input_img_path = ""

                    if self.input_dir_var and self.input_dir_var.get():
                        input_dir = self.input_dir_var.get()

                        # In the input folder, the image has the same name as the one in the output folder
                        # but with a "_COLOR_SPACE" suffix and a ".jpg" extension
                        # Example: for "1_1.png", look for "1_1_RGB.jpg", "1_1_LAB.jpg", etc.

                        # List of possible color spaces
                        color_spaces = ["RGB", "YUV", "HSV", "LAB", "HLS"]

                        # Try all possible color spaces
                        for cs in color_spaces:
                            possible_path = os.path.join(input_dir, f"{output_name}_{cs}.jpg")
                            if os.path.exists(possible_path):
                                input_img_path = possible_path
                                break

                        # If no variant is found, try without color space
                        if not input_img_path:
                            fallback_path = os.path.join(input_dir, f"{output_name}.jpg")
                            if os.path.exists(fallback_path):
                                input_img_path = fallback_path

                    # Calculate the dimensions of the popup window
                    screen_w, screen_h = (
                        self.winfo_screenwidth(),
                        self.winfo_screenheight(),
                    )
                    max_w, max_h = int(screen_w * 0.9), int(screen_h * 0.9)

                    # Resize the output image if necessary
                    if output_orig_img.width > max_w / 2 or output_orig_img.height > max_h:
                        ratio = min(
                            max_w / 2 / output_orig_img.width,
                            max_h / output_orig_img.height,
                        )
                        output_disp_img = output_orig_img.resize(
                            (
                                int(output_orig_img.width * ratio),
                                int(output_orig_img.height * ratio),
                            ),
                            Image.LANCZOS,
                        )
                    else:
                        output_disp_img = output_orig_img

                    # Create the popup window
                    popup = tk.Toplevel(self)
                    popup.title(f"Preview: {output_basename}")
                    popup.geometry(
                        f"{min(output_disp_img.width * 2, screen_w)}x{min(output_disp_img.height, screen_h)}+0+0"
                    )

                    # Create a frame to contain the two images
                    frame = ttk.Frame(popup)
                    frame.pack(expand=True, fill="both")

                    # Place the input image on the left
                    left_frame = ttk.Frame(frame)
                    left_frame.pack(side="left", expand=True, fill="both")
                    input_label = ttk.Label(left_frame)
                    input_label.pack(expand=True, fill="both")

                    # Place the output image on the right
                    right_frame = ttk.Frame(frame)
                    right_frame.pack(side="right", expand=True, fill="both")
                    output_label = ttk.Label(right_frame)
                    output_label.pack(expand=True, fill="both")

                    # Display the output image
                    output_img_tk = ImageTk.PhotoImage(output_disp_img)
                    output_label.config(image=output_img_tk)
                    output_label.image = output_img_tk

                    # Try to display the input image
                    try:
                        # Determine the color space from the filename
                        input_filename = os.path.basename(input_img_path)
                        input_name, _ = os.path.splitext(input_filename)

                        # Read the image with OpenCV
                        cv_img = cv2.imread(input_img_path)

                        # Convert the image according to the color space identified in the filename
                        if "_RGB" in input_name.upper():
                            # Already in RGB, just invert BGR to RGB
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        elif "_LAB" in input_name.upper():
                            # Convert LAB to RGB
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_LAB2RGB)
                        elif "_HSV" in input_name.upper():
                            # Convert HSV to RGB
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_HSV2RGB)
                        elif "_HLS" in input_name.upper():
                            # Convert HLS to RGB
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_HLS2RGB)
                        elif "_YUV" in input_name.upper():
                            # Convert YUV to RGB
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_YUV2RGB)
                        else:
                            # If no color space is specified, assume BGR
                            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                        # Convert the OpenCV image to a PIL image
                        input_orig_img = Image.fromarray(cv_img_rgb)

                        # Resize the input image to have the same aspect ratio as the output image
                        if input_orig_img.width > max_w / 2 or input_orig_img.height > max_h:
                            ratio = min(
                                max_w / 2 / input_orig_img.width,
                                max_h / input_orig_img.height,
                            )
                            input_disp_img = input_orig_img.resize(
                                (
                                    int(input_orig_img.width * ratio),
                                    int(input_orig_img.height * ratio),
                                ),
                                Image.LANCZOS,
                            )
                        else:
                            input_disp_img = input_orig_img

                        input_img_tk = ImageTk.PhotoImage(input_disp_img)
                        input_label.config(image=input_img_tk)
                        input_label.image = input_img_tk

                        # Add labels to distinguish the images
                        ttk.Label(left_frame, text="Input Image").pack(side="top")
                        ttk.Label(right_frame, text="Segmented Image").pack(side="top")
                    except FileNotFoundError:
                        # If the input image does not exist, display a message
                        input_label.config(text=f"Input image not found:\n{input_img_path}")

                    self._img_popup = popup
                except Exception as e:
                    msg = f"Error displaying images: {e}"
                    logger.exception(msg)
            sel.annotation.set_visible(False)

        cursor.connect("add", show_img_on_click)

        self.stats_canvas = FigureCanvasTkAgg(fig, master=self)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill="both", expand=True)
