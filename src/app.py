"""Entry point for the application."""

import tkinter as tk

from src.gui import App


def main() -> None:
    """Start the application."""
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
