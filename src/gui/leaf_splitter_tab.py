"""Tab for image splitting functionality."""

import threading
import tkinter as tk
import typing
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from src import config
from src.image_processing.batch_processor import BatchImageProcessor


class ImageSplitterTab(ttk.Frame):
    """Tab containing the image splitting interface."""

    def __init__(self, parent: tk.Widget) -> None:
        """Initialize the ImageSplitterTab."""
        super().__init__(parent)
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.color_space = tk.StringVar(value="RGB")
        self.dpi_value = tk.StringVar(value=str(config.DPI))

        self._setup_layout()
        self._create_widgets()

    def _setup_layout(self) -> None:
        """Set up the main layout with left and right frames."""
        self.left_frame = tk.Frame(self, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_widgets(self) -> None:
        """Create all widgets for the tab."""
        self._add_directory_selector("Input Directory:", self.input_dir, self.browse_input)
        self._add_directory_selector("Output Directory:", self.output_dir, self.browse_output)

        # Color space dropdown
        tk.Label(self.left_frame, text="Color Space:").pack(pady=(10, 0))
        color_options = ["RGB", "YUV", "HSV", "LAB", "HLS"]
        color_menu = ttk.Combobox(
            self.left_frame,
            textvariable=self.color_space,
            values=color_options,
            state="readonly",
            width=20,
        )
        color_menu.pack(pady=2)

        # DPI entry
        tk.Label(self.left_frame, text="Scanner DPI:").pack(pady=(10, 0))
        dpi_frame = tk.Frame(self.left_frame)
        dpi_frame.pack(pady=2)

        dpi_entry = tk.Entry(dpi_frame, textvariable=self.dpi_value, width=10)
        dpi_entry.pack(side=tk.LEFT, padx=2)

        # Validate function to ensure only integers are entered
        def validate_dpi(action: str, value_if_allowed: str) -> bool:
            if action == "1":  # insert
                return bool(value_if_allowed.isdigit() and int(value_if_allowed) > 0)
            return True

        vcmd = (self.register(validate_dpi), "%d", "%P")
        dpi_entry.config(validate="key", validatecommand=vcmd)

        # Update DPI value when changed
        def update_dpi(*_args) -> None:  # noqa: ANN002
            try:
                new_dpi = int(self.dpi_value.get())
                if new_dpi > 0:
                    config.DPI = new_dpi
            except ValueError:
                pass

        self.dpi_value.trace_add("write", update_dpi)

        tk.Button(
            self.left_frame,
            text="Start Splitting",
            command=self.run_split,
            width=22,
        ).pack(pady=18)

        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(self.left_frame, orient="horizontal", length=220, mode="determinate")
        self.progress.pack(pady=(0, 10))
        self.progress.pack_forget()

        # Stats zone
        self.stats_label = tk.Label(self.right_frame, text="", font=("Arial", 12), justify="left", anchor="nw")
        self.stats_label.pack(pady=10, anchor="nw")
        self.stats_canvas = None
        self._img_popup = None

    def _add_directory_selector(self, label: str, var: tk.StringVar, command: typing.Callable) -> None:
        """Add a directory selection widget with label and browse button."""
        tk.Label(self.left_frame, text=label).pack(pady=(10, 0))
        entry_frame = tk.Frame(self.left_frame)
        entry_frame.pack(pady=2)
        tk.Entry(entry_frame, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
        tk.Button(entry_frame, text="Browse", command=command).pack(side=tk.LEFT)

    def browse_input(self) -> None:
        """Open file dialog to select input directory."""
        path = filedialog.askdirectory(title="Choose Input Directory")
        if path:
            self.input_dir.set(path)

    def browse_output(self) -> None:
        """Open file dialog to select output directory."""
        path = filedialog.askdirectory(title="Choose Output Directory")
        if path:
            self.output_dir.set(path)

    def run_split(self) -> None:
        """Start the image splitting process in a separate thread."""

        def process() -> None:
            processor = BatchImageProcessor(
                self.input_dir.get(),
                self.output_dir.get(),
                color_space=self.color_space.get(),
            )

            is_valid, error_msg = processor.validate_directories()
            if not is_valid:
                self._show_error(error_msg)
                return

            images = processor.find_images()
            if not images:
                self.after(0, self.progress.pack_forget)
                self._show_warning("No images found in the input directory.")
                return

            self.after(0, self.progress.pack)
            self.after(0, lambda: self.progress.config(maximum=len(images), value=0))

            def update_progress(current: int) -> None:
                self.after(0, lambda: self.progress.config(value=current))

            processor.process_images(progress_callback=update_progress)
            self.after(0, self.progress.pack_forget)

            # Get stats about processed images
            widths, heights, output_images = processor.get_output_stats()
            self.after(0, lambda: self.show_stats_from_data(widths, heights, output_images))

        threading.Thread(target=process).start()

    def _show_error(self, msg: str) -> None:
        """Show error message dialog."""
        self.after(0, lambda: messagebox.showerror("Error", msg))

    def _show_warning(self, msg: str) -> None:
        """Show warning message dialog."""
        self.after(0, lambda: messagebox.showwarning("Warning", msg))

    def show_stats_from_data(self, widths: list, heights: list, image_paths: list) -> None:
        """Display statistics and scatter plot of processed images."""
        if not widths or not heights:
            self.stats_label.config(text="Impossible to read image dimensions.")
            if self.stats_canvas:
                self.stats_canvas.get_tk_widget().pack_forget()
            return

        stats = (
            f"Number of images: {len(widths)}\n"
            f"Average width: {np.mean(widths):.1f} mm (min: {np.min(widths):.1f}, max: {np.max(widths):.1f})\n"
            f"Average height: {np.mean(heights):.1f} mm (min: {np.min(heights):.1f}, max: {np.max(heights):.1f})"
        )
        self.stats_label.config(text=stats)

        # Remove old chart
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
        scatter = ax_scatter.scatter(widths, heights, alpha=0.8, color="blue")
        ax_scatter.set_xlabel("Width (mm)")
        ax_scatter.set_ylabel("Height (mm)")
        ax_scatter.set_xlim(left=0)
        ax_scatter.set_ylim(bottom=0)

        # Histograms
        bins = min(int(len(widths) / 2), 30)  # More bins for better granularity
        if bins < 10:  # Ensure at least 10 bins if we have enough data points  # noqa: PLR2004
            bins = max(10, len(widths) // 2)

        ax_histx.hist(widths, bins=bins, alpha=0.8, color="blue", edgecolor="black")
        ax_histy.hist(
            heights,
            bins=bins,
            alpha=0.8,
            color="blue",
            edgecolor="black",
            orientation="horizontal",
        )

        # Adjust the layout to minimize empty space
        plt.subplots_adjust(top=0.92, right=0.95)

        cursor = mplcursors.cursor(scatter, hover=False)

        def show_img_on_click(sel: mplcursors.Selection) -> None:
            """Show image preview when clicking a point in the scatter plot."""
            idx = sel.index
            if self._img_popup is not None and self._img_popup.winfo_exists():
                self._img_popup.destroy()
            if 0 <= idx < len(image_paths):
                try:
                    img = Image.open(image_paths[idx])
                    orig_img = img.copy()
                    screen_w, screen_h = (
                        self.winfo_screenwidth(),
                        self.winfo_screenheight(),
                    )
                    max_w, max_h = int(screen_w * 0.9), int(screen_h * 0.9)
                    if orig_img.width > max_w or orig_img.height > max_h:
                        ratio = min(max_w / orig_img.width, max_h / orig_img.height)
                        disp_img = orig_img.resize(
                            (int(orig_img.width * ratio), int(orig_img.height * ratio)),
                            Image.LANCZOS,
                        )
                    else:
                        disp_img = orig_img

                    popup = tk.Toplevel(self)
                    popup.title(f"Preview: {image_paths[idx].name}")
                    popup.geometry(f"{min(disp_img.width * 2, screen_w)}x{min(disp_img.height, screen_h)}+0+0")

                    label = tk.Label(popup)
                    label.pack(expand=True, fill="both")
                    img_tk = ImageTk.PhotoImage(disp_img)
                    label.config(image=img_tk)
                    label.image = img_tk
                    self._img_popup = popup
                except OSError:
                    pass
            sel.annotation.set_visible(False)

        cursor.connect("add", show_img_on_click)

        self.stats_canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill="both", expand=True)
