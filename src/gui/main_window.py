"""Main window of the image splitter application."""

import tkinter as tk
from tkinter import ttk

from src.gui.class_config_tab import ClassConfigTab
from src.gui.leaf_segmenter_tab import LeafSegmenterTab
from src.gui.leaf_splitter_tab import ImageSplitterTab


class App:
    """Main window of the application."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main application window."""
        self.root = root
        self.root.title("MISSION R&D")
        self.root.geometry("1000x600")
        self.root.resizable(width=True, height=True)

        self._setup_notebook()

    def _setup_notebook(self) -> None:
        """Set up the main notebook with tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # First tab: image splitting
        self.splitter_tab = ImageSplitterTab(self.notebook)
        self.notebook.add(self.splitter_tab, text="Image Splitter")

        # Second tab: segmentation
        self.segmenter_tab = LeafSegmenterTab(self.notebook)
        self.notebook.add(self.segmenter_tab, text="Image Segmenter")

        # Third tab: class configuration
        self.class_config_tab = ClassConfigTab(self.notebook)
        self.notebook.add(self.class_config_tab, text="Class Configuration")

        # Synchronize: if splitter output_dir is set, set segmenter input_dir by default
        def sync_segmenter_input(*_args) -> None:  # noqa: ANN002
            out_dir = self.splitter_tab.output_dir.get()
            if out_dir:
                self.segmenter_tab.input_dir.set(out_dir)

        # Sync DPI values between tabs
        def sync_dpi_value(*_args) -> None:  # noqa: ANN002
            dpi_value = self.splitter_tab.dpi_value.get()
            if dpi_value:
                self.segmenter_tab.dpi_value.set(dpi_value)

        # Initial sync if already set
        sync_segmenter_input()
        sync_dpi_value()

        # Trace changes to splitter output_dir
        self.splitter_tab.output_dir.trace_add("write", lambda *_args: sync_segmenter_input())

        # Trace changes to splitter DPI value
        self.splitter_tab.dpi_value.trace_add("write", lambda *_args: sync_dpi_value())


def main() -> None:
    """Run the main application."""
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
