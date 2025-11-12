"""Tab for configuring class names and colors for segmentation."""

import json
import logging
import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.utils.color_map_utils import get_color_map


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassConfigTab(ttk.Frame):
    """Tab for configuring segmentation classes and their colors."""

    def __init__(self, parent: tk.Widget) -> None:
        """Initialize the class configuration tab."""
        super().__init__(parent)
        # Use the same path as in color_map_utils.py
        self.color_map_path = Path(__file__).parent.parent / "color_map.json"
        self.class_entries = []  # Store references to entry widgets
        self.update_scheduled = False  # To avoid too many updates
        self.setup_ui()

        # Longer loading delay to ensure the interface is fully rendered
        self.after(200, self.delayed_load)

    def delayed_load(self) -> None:
        """Delayed loading with multiple interface refreshes."""
        self.load_color_map()
        # Schedule multiple preview refreshes
        self.after(100, self.refresh_preview)
        self.after(300, self.refresh_preview)
        self.after(500, self.refresh_preview)

    def refresh_preview(self) -> None:
        """Force a complete update of the preview and interface."""
        self.update_preview()
        self.update_idletasks()  # Force processing of pending interface tasks

    def setup_ui(self) -> None:
        """Set up the main UI components."""
        # Split into two frames
        self.left_frame = ttk.Frame(self, width=400)  # Wider left frame
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title and instructions
        ttk.Label(self.left_frame, text="Class Configuration", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 10)
        )

        ttk.Label(
            self.left_frame,
            text="Define the names and values of segmentation classes.\n"
            "Each class must have a unique name and a value between 0-255.",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 15))

        # Frame for class entries with scrollbar
        self.entries_frame = ttk.Frame(self.left_frame)
        self.entries_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        self.canvas = tk.Canvas(self.entries_frame, width=380)  # Set initial width
        self.scrollbar = ttk.Scrollbar(self.entries_frame, orient="vertical", command=self.canvas.yview)
        self.classes_frame = ttk.Frame(self.canvas)

        self.classes_frame.bind(
            "<Configure>",
            lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.classes_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add/Save buttons
        self.buttons_frame = ttk.Frame(self.left_frame)
        self.buttons_frame.pack(fill=tk.X, pady=10)

        self.add_button = ttk.Button(self.buttons_frame, text="Add Class", command=self.add_empty_class)
        self.add_button.pack(side=tk.LEFT, padx=(0, 5))

        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.save_color_map)
        self.save_button.pack(side=tk.RIGHT)

        # Preview area
        ttk.Label(self.right_frame, text="Class Preview", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        self.preview_canvas = tk.Canvas(
            self.right_frame,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray",
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Refresh preview when canvas is resized
        self.preview_canvas.bind("<Configure>", lambda _: self.after(100, self.update_preview))

    def load_color_map(self) -> None:
        """Load the existing color map from the JSON file."""
        try:
            self.color_map = get_color_map()

            # Clear existing entries
            for widget in self.classes_frame.winfo_children():
                widget.destroy()
            self.class_entries = []

            # Add entries for each class in the color map
            for class_name, value in self.color_map.items():
                self.add_class_entry(class_name, str(value))

            # If no classes were loaded, add an empty one
            if not self.class_entries:
                self.add_empty_class()

            # Ensure width is sufficient to display all elements
            self.classes_frame.update_idletasks()
            min_width = (
                max(w.winfo_width() for w in self.classes_frame.winfo_children())
                if self.classes_frame.winfo_children()
                else 350
            )
            self.canvas.config(width=min_width)

            # Force interface refresh
            self.update()
            self.update_idletasks()
            self.update_preview()

        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror(
                "Error Loading Color Map",
                f"An error occurred while loading the color map: {e!s}",
            )
            self.color_map = {}
            self.add_empty_class()

    def add_empty_class(self) -> None:
        """Add an empty class entry row."""
        # Generate a random value between 0 and 255 that's not already used
        used_values = list(self.color_map.values())
        available_values = [i for i in range(64, 256, 64) if i not in used_values]

        # If all standard values are used, find any available value
        if not available_values:
            available_values = [i for i in range(1, 256) if i not in used_values]

        # If somehow all 255 values are used (unlikely), just use 0
        value = random.choice(available_values) if available_values else 0  # noqa: S311

        msg = f"Adding new class with value: {value}"
        logger.info(msg)
        self.add_class_entry("", str(value))

        # Update preview immediately and force refresh
        self.update_preview()
        self.update_idletasks()
        msg = f"After adding: {len(self.class_entries)} entries"
        logger.info(msg)

    def add_class_entry(self, class_name: str = "", class_value: str = "") -> tuple[tk.StringVar, tk.StringVar]:
        """Add a new class entry row to the configuration."""
        row_frame = tk.Frame(self.classes_frame, padx=3, pady=5, relief=tk.GROOVE, bd=1)
        row_frame.pack(fill=tk.X, padx=5, pady=3, ipady=3)

        # Class name entry
        name_label = tk.Label(row_frame, text="Class:")
        name_label.pack(side=tk.LEFT, padx=(0, 5))
        name_var = tk.StringVar(value=class_name)
        name_entry = tk.Entry(row_frame, textvariable=name_var, width=10)
        name_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Add trace to update preview when name changes
        name_var.trace_add("write", lambda *_: self.update_preview())

        # Class value entry
        value_label = tk.Label(row_frame, text="Value:")
        value_label.pack(side=tk.LEFT, padx=(0, 5))
        value_var = tk.StringVar(value=class_value)
        value_entry = tk.Entry(row_frame, textvariable=value_var, width=5)
        value_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Add trace to update preview when value changes
        value_var.trace_add("write", lambda *_: self.update_preview())

        # Remove button - improved visibility
        remove_button = tk.Button(
            row_frame,
            text="Remove",
            bg="#ffcccc",
            command=lambda: self.remove_class_entry(row_frame),
            width=8,
            padx=5,
        )
        remove_button.pack(side=tk.RIGHT, padx=(10, 0))

        # Store references
        self.class_entries.append((name_var, value_var, row_frame))

        # Mettre à jour l'aperçu immédiatement
        self.update_preview()

        return name_var, value_var

    def validate_value(self, new_value: str) -> bool:
        """Validate that the value entry contains a number between 0-255."""
        if not new_value:
            return True
        try:
            value = int(new_value)
        except ValueError:
            return False
        else:
            return 0 <= value <= 255  # noqa: PLR2004

    def remove_class_entry(self, row_frame: tk.Frame) -> None:
        """Remove a class entry row."""
        # Find and remove from our tracked lists
        for i, (_, _, frame) in enumerate(self.class_entries):
            if frame == row_frame:
                self.class_entries.pop(i)
                break

        # Destroy the frame and all its children
        row_frame.destroy()
        self.update_preview()

    def update_preview(self) -> None:
        """Update the color preview canvas with debouncing to avoid too frequent updates."""
        # If an update is already scheduled, do nothing
        if self.update_scheduled:
            return

        # Mark that an update is scheduled
        self.update_scheduled = True

        # Schedule the actual update after a short delay (100ms)
        self.after(100, self._do_update_preview)

    def _do_update_preview(self) -> None:
        """Actual implementation of the preview update."""
        # Reset flag to allow future updates
        self.update_scheduled = False

        # Force complete refresh
        self.update_idletasks()

        self.preview_canvas.delete("all")

        # Ensure canvas has valid dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        # If canvas doesn't have valid dimensions yet, use default values
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300

        # Collect current classes and values
        classes = []
        class_values = []

        for name_var, value_var, _ in self.class_entries:
            class_name = name_var.get().strip()

            # Get the value
            try:
                value = int(value_var.get())
            except (ValueError, TypeError):
                value = 0

            # Use temporary label if name is empty
            display_name = class_name if class_name else "(Unnamed)"
            classes.append(display_name)
            class_values.append((display_name, value))

        # If no valid classes, show a message
        if not classes:
            self.preview_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="No classes defined",
                font=("Arial", 12),
            )
            return

        # Draw color blocks with appropriate sizing
        num_classes = len(class_values)
        block_height = min(40, (canvas_height - 20) / max(num_classes, 1))

        y_pos = 10
        for class_name, value in class_values:
            # Draw the color block
            color = f"#{value:02x}{value:02x}{value:02x}"
            self.preview_canvas.create_rectangle(
                10,
                y_pos,
                canvas_width - 10,
                y_pos + block_height,
                fill=color,
                outline="black",
            )

            # Draw text (white or black depending on background)
            text_color = "white" if value < 128 else "black"  # noqa: PLR2004
            self.preview_canvas.create_text(
                20,
                y_pos + block_height / 2,
                text=f"{class_name} ({value})",
                fill=text_color,
                anchor="w",
                font=("Arial", 10),
            )

            y_pos += block_height + 5

    def save_color_map(self) -> None:
        """Save the current classes and values to the color_map.json file."""
        new_color_map = {}
        duplicate_values = set()
        empty_names = False

        # Collect all class entries
        for name_var, value_var, _ in self.class_entries:
            class_name = name_var.get().strip()

            # Skip empty class names
            if not class_name:
                empty_names = True
                continue

            # Get the value
            try:
                value = int(value_var.get())
            except (ValueError, TypeError):
                continue

            # Check for duplicate values
            if value in new_color_map.values():
                duplicate_values.add(value)

            new_color_map[class_name] = value

        # Validate before saving
        if empty_names:
            messagebox.showwarning("Empty Names", "Some classes have empty names and will be skipped.")

        if duplicate_values:
            messagebox.showerror(
                "Duplicate Values",
                f"Found duplicate values: {', '.join(map(str, duplicate_values))}.",
                "Each class must have a unique value.",
            )
            return

        if not new_color_map:
            messagebox.showerror(
                "No Classes",
                "No valid classes to save. Please add at least one class with a name and value.",
            )
            return

        # Save to the file
        try:
            with Path(self.color_map_path).open("w") as f:
                json.dump(new_color_map, f, indent=4)
            messagebox.showinfo(
                "Success",
                f"Successfully saved {len(new_color_map)} classes to {self.color_map_path}",
            )

            # Update our internal color map
            self.color_map = new_color_map
        except OSError as e:
            messagebox.showerror("Error Saving", f"An error occurred while saving: {e!s}")
