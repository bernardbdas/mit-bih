import os
import datetime
import matplotlib.pyplot as plt

def save_plot(filename, fig=None, base_dir=None, caption=None):
    """
    Saves a matplotlib figure with a timestamp in the active directory's assets/ subfolder,
    and optionally saves a sidecar text file containing the figure's caption.
    
    Args:
        filename (str): Name of the file (e.g. 'confusion_matrix' or 'loss_curve.png').
        fig (matplotlib.figure.Figure, optional): The figure to save. If None, saves plt.gcf().
        base_dir (str, optional): Target parent folder. Defaults to the current working directory.
        caption (str, optional): Text caption describing the figure. Saved to a sidecar .txt file.
        
    Returns:
        str: Absolute path to the saved figure image.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    assets_dir = os.path.join(base_dir, "assets", today_str)
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
