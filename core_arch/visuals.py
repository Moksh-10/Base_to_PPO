import os
import numpy as np
import matplotlib.pyplot as plt


out_dir = os.path.join(os.path.dirname(__file__), 'out')

def _ensure_out():
    os.makedirs(out_dir, exist_ok=True)

def save_matrix_heatmap(mat: np.ndarray, title: str, filename: str, xlabel: str = '', ylabel: str = ''):
    _ensure_out()
    plt.figure()
    plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"saved; {path}")


def save_atnn_heads_grid(wei: np.ndarray, filename: str, title_prefix: str = "head"):
    _ensure_out()
    bs, nh, sl, dim = wei.shape
    cols = min(4, nh)
    rows = (nh + cols - 1) // cols
    plt.figure(figsize=(3*cols, 3*rows))
    for h in range(nh):
        ax = plt.subplot(rows, cols, h+1)
        ax.imshow(wei[0, h], aspect='auto')
        ax.set_title(f"{title_prefix} {h}")
        ax.set_xlabel('key pos')
        ax.set_ylabel('query pos')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"saved: {path}")




