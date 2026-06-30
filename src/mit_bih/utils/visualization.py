"""
Visualization utility helpers.

This module provides helper functions to save matplotlib plots and figures
directly into the project's assets/plots directory.
"""

import os
import datetime
import matplotlib.pyplot as plt


def save_plot(filename: str, fig: plt.Figure | None = None, base_dir: str | None = None, caption: str | None = None) -> str:
    """
    Saves a matplotlib figure with a timestamp in the project's assets/plots/ subfolder,
    and optionally saves a sidecar text file containing the figure's caption.

    Args:
        filename: Name of the file (e.g. 'confusion_matrix' or 'loss_curve.png').
        fig: The figure to save. If None, saves the active figure (plt.gcf()).
        base_dir: Target parent folder. Defaults to the current working directory.
        caption: Text caption describing the figure. Saved to a sidecar .txt file.

    Returns:
        Absolute path to the saved figure image.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    # Save directly to assets/plots as requested
    assets_dir = os.path.join(base_dir, "assets", "plots")
    os.makedirs(assets_dir, exist_ok=True)

    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{name}_{timestamp}{ext}"
    save_path = os.path.join(assets_dir, save_filename)

    if fig is None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    print(f"Saved figure to: {save_path}")
    
    if caption:
        caption_path = os.path.splitext(save_path)[0] + ".txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"Saved figure caption sidecar to: {caption_path}")

    return save_path
