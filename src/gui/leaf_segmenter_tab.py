"""Tab for image segmentation functionality."""

import logging
import os
import threading
import time
import tkinter as tk
import typing
from tkinter import filedialog, ttk

from src import config
from src.gui.segmentation_stats_frame import SegmentationStatsFrame
from src.image_processing.leaf_segmenter import LeafSegmenter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeafSegmenterTab(ttk.Frame):
    """Tab containing the leaf segmentation interface."""

    def __init__(self, parent: tk.Widget) -> None:
        """Initialize the LeafSegmenterTab."""
        super().__init__(parent)
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.dpi_value = tk.StringVar(value=str(config.DPI))
        self._setup_layout()
        self._create_widgets()
        # Trace output_dir to update stats automatically
        self.output_dir.trace_add("write", self._on_output_dir_change)

    def _setup_layout(self) -> None:
        self.left_frame = tk.Frame(self, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_widgets(self) -> None:
        self._add_directory_selector("Input Directory:", self.input_dir, self.browse_input)
        self._add_directory_selector("Output Directory:", self.output_dir, self.browse_output)
        self._add_file_selector("Model File:", self.model_path, self.browse_model)

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
        def update_dpi(*args: typing.Any) -> None:  # noqa: ANN401, ARG001
            try:
                new_dpi = int(self.dpi_value.get())
                if new_dpi > 0:
                    config.DPI = new_dpi
            except ValueError:
                pass

        self.dpi_value.trace_add("write", update_dpi)

        tk.Button(
            self.left_frame,
            text="Start Segmentation",
            command=self.run_segmentation,
            width=22,
        ).pack(pady=18)

        # Frame for progress bar and label
        self.progress_frame = tk.Frame(self.left_frame)
        self.progress_frame.pack(pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=220, mode="determinate")
        self.progress.pack(side=tk.TOP)

        # Label for displaying progress
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Arial", 12))
        self.progress_label.pack(side=tk.TOP, pady=(2, 0))

        # Hide progress frame by default
        self.progress_frame.pack_forget()
        self.status_label = tk.Label(self.right_frame, text="", font=("Arial", 12), justify="left", anchor="nw")
        self.status_label.pack(pady=10, anchor="nw")

        self.stats_frame = SegmentationStatsFrame(self.right_frame, self.output_dir, self.input_dir)
        self.stats_frame.pack(fill="both", expand=True, pady=10)

    def _add_directory_selector(self, label: str, var: tk.StringVar, command: typing.Callable) -> None:
        tk.Label(self.left_frame, text=label).pack(pady=(10, 0))
        entry_frame = tk.Frame(self.left_frame)
        entry_frame.pack(pady=2)
        tk.Entry(entry_frame, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
        tk.Button(entry_frame, text="Browse", command=command).pack(side=tk.LEFT)

    def _add_file_selector(self, label: str, var: tk.StringVar, command: typing.Callable) -> None:
        tk.Label(self.left_frame, text=label).pack(pady=(10, 0))
        entry_frame = tk.Frame(self.left_frame)
        entry_frame.pack(pady=2)
        tk.Entry(entry_frame, textvariable=var, width=24).pack(side=tk.LEFT, padx=2)
        tk.Button(entry_frame, text="Browse", command=command).pack(side=tk.LEFT)

    def browse_input(self) -> None:
        """Open a file dialog to select the input directory."""
        path = filedialog.askdirectory(title="Choose Input Directory")
        if path:
            self.input_dir.set(path)

    def browse_output(self) -> None:
        """Open a file dialog to select the output directory."""
        path = filedialog.askdirectory(title="Choose Output Directory")
        if path:
            self.output_dir.set(path)
            # No need to call update here, handled by trace

    def browse_model(self) -> None:
        """Open a file dialog to select the model file."""
        path = filedialog.askopenfilename(title="Choose Model File", filetypes=[("Model files", "*.*")])
        if path:
            self.model_path.set(path)

    def run_segmentation(self) -> None:
        """Run the leaf segmentation process in a separate thread."""

        def process() -> None:
            # Reset progress bar and label
            self.progress.config(value=0)

            try:
                start_time = time.time()
                input_dir = self.input_dir.get()
                output_dir = self.output_dir.get()

                # Check if directories exist and have proper permissions
                if not os.path.exists(input_dir):
                    self._show_status(f"Error: Input directory '{input_dir}' does not exist.")
                    return

                # Ensure output directory exists or can be created
                os.makedirs(output_dir, exist_ok=True)

                # Check input files
                input_images = [
                    f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                ]
                total_images = len(input_images)

                if total_images == 0:
                    self._show_status("No images found in the input directory.")
                    return

                # Check model file
                model_path = self.model_path.get()
                if model_path and not os.path.exists(model_path):
                    self._show_status(f"Error: Model file '{model_path}' does not exist.")
                    return

                # Create segmenter object without callback (we'll manage progress ourselves)
                try:
                    segmenter = LeafSegmenter(model_path)
                except (OSError, ValueError, RuntimeError) as e:
                    self._show_status(f"Error loading model: {e}")
                    return

                # Start progress monitoring thread
                stop_monitoring = threading.Event()
                monitor_thread = threading.Thread(
                    target=self._monitor_progress,
                    args=(output_dir, total_images, stop_monitoring),
                    daemon=True,
                )
                monitor_thread.start()

                # Execute segmentation
                try:
                    segmenter.segmenter(input_dir, output_dir)
                    segmenter.make_bilan()
                except (OSError, ValueError, RuntimeError) as e:
                    stop_monitoring.set()
                    monitor_thread.join(timeout=1.0)
                    self._show_status(f"Error during segmentation: {e}")
                    return

                # Stop monitoring and finalize progress bar
                stop_monitoring.set()
                monitor_thread.join(timeout=1.0)

                # Calculate total elapsed time
                total_elapsed_time = time.time() - start_time

                # Format that adapts to total time (hours, minutes, seconds)
                time_format = ""
                if total_elapsed_time >= 3600:  # More than an hour  # noqa: PLR2004
                    hours = int(total_elapsed_time // 3600)
                    minutes = int((total_elapsed_time % 3600) // 60)
                    seconds = int(total_elapsed_time % 60)
                    time_format = f"Total time: {hours}h {minutes}m {seconds}s"
                elif total_elapsed_time >= 60:  # More than a minute  # noqa: PLR2004
                    minutes = int(total_elapsed_time // 60)
                    seconds = int(total_elapsed_time % 60)
                    time_format = f"Total time: {minutes}m {seconds}s"
                else:  # Less than a minute
                    seconds = int(total_elapsed_time)
                    time_format = f"Total time: {seconds}s"

                # Update progress bar to 100%
                self.after(0, lambda: self.progress.config(value=100))
                self.after(
                    0,
                    lambda: self.progress_label.config(
                        text=f"Images processed: {total_images}/{total_images} (100%)\n{time_format}"
                    ),
                )

                self._show_status(f"Segmentation completed. {time_format}")
                # Update stats in main thread
                self.after(0, self._update_stats)
            except (OSError, ValueError, RuntimeError) as e:
                msg = f"Unexpected error during segmentation: {e}"
                self._show_status(msg)
            finally:
                # Hide progress frame once done
                self.after(500, lambda: self.progress_frame.pack_forget())

        # Show and configure progress bar before starting
        self.progress.config(mode="determinate", maximum=100, value=0)
        self.progress_label.config(text="Please wait...")
        self.progress_frame.pack(pady=(0, 10))

        # Start processing in a separate thread
        threading.Thread(target=process, daemon=True).start()

    def _on_output_dir_change(self) -> None:
        self._update_stats()

    def _update_stats(self) -> None:
        if hasattr(self, "stats_frame") and hasattr(self.stats_frame, "update_stats"):
            self.stats_frame.update_stats()

    def _monitor_progress(self, output_dir: str, total_images: int, stop_event: threading.Event) -> None:
        """
        Monitor the output directory to track segmentation progress.

        Args:
            output_dir (str): Path to the output directory
            total_images (int): Total number of images to process
            stop_event (threading.Event): Event to stop monitoring

        """
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        last_count = 0
        start_time = time.time()

        while not stop_event.is_set():
            try:
                # Check for segmented images (with and without the _Simple_Segmentation suffix)
                try:
                    output_files = [
                        f
                        for f in os.listdir(output_dir)
                        if (f.lower().endswith((".png", ".tif", ".tiff")) and not f.endswith(".csv"))
                    ]
                    processed_count = len(output_files)
                except Exception as e:
                    msg = f"Error accessing output directory: {e}"
                    logger.exception(msg)
                    self.after(0, lambda: self._show_status("Error monitoring progress"))
                    return

                # If the number has changed, update the progress bar
                if processed_count != last_count:
                    # Limit to total_images to prevent overflow
                    processed_count = min(processed_count, total_images)
                    percentage = int((processed_count / total_images) * 100)
                    self.after(0, lambda p=percentage: self.progress.config(value=p))

                    # Calculate remaining time estimate if at least one image has been processed
                    if processed_count > 0:
                        elapsed_time = time.time() - start_time
                        time_per_image = elapsed_time / processed_count
                        remaining_images = total_images - processed_count
                        remaining_time = time_per_image * remaining_images

                        # Format that adapts to remaining time (hours, minutes, seconds)
                        time_format = ""
                        if remaining_time >= 3600:  # More than an hour  # noqa: PLR2004
                            hours = int(remaining_time // 3600)
                            minutes = int((remaining_time % 3600) // 60)
                            seconds = int(remaining_time % 60)
                            time_format = f"Estimated time remaining: {hours}h {minutes}m {seconds}s"
                        elif remaining_time >= 60:  # More than a minute  # noqa: PLR2004
                            minutes = int(remaining_time // 60)
                            seconds = int(remaining_time % 60)
                            time_format = f"Estimated time remaining: {minutes}m {seconds}s"
                        else:  # Less than a minute
                            seconds = int(remaining_time)
                            time_format = f"Estimated time remaining: {seconds}s"

                        # Progress text on the first line
                        progress_text = f"Progress: {processed_count}/{total_images} ({percentage}%)"
                        # Time estimation on a second line
                        status_text = f"{progress_text}\n{time_format}"
                    else:
                        status_text = f"Progress: {processed_count}/{total_images} ({percentage}%)"

                    self.after(0, lambda t=status_text: self.progress_label.config(text=t))
                    last_count = processed_count

                # Short pause before the next check
                time.sleep(0.5)

            except Exception as e:  # noqa: PERF203
                msg = f"Error monitoring progress: {e}"
                logger.exception(msg)
                time.sleep(1)

    def _show_status(self, msg: str) -> None:
        self.status_label.config(text=msg)
