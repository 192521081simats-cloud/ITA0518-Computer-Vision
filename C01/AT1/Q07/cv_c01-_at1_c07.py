"""
Computer Vision - AT1 Short Answer Test
Question 7: A grayscale image is converted into a digital image. Describe
the role of quantization in this process and its impact on image detail.

Demo: the same image quantized to different bit depths, with an MSE metric
to quantify the resulting loss of detail (false contouring / banding).

Run:  python3 q7.py
Requires: numpy, matplotlib
Output: q7_quantization_effect.png
"""

import numpy as np
import matplotlib.pyplot as plt


def q7_quantization_effect():
    size = 256
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    # smooth gradient with a soft circular object -> simulates a grayscale photo
    original = (0.6 * xx + 0.4 * np.exp(-((xx - 0.5)**2 + (yy - 0.5)**2) / 0.05))
    original = np.clip(original, 0, 1)

    def quantize(img, levels):
        step = 1.0 / (levels - 1)
        return np.round(img / step) * step

    bit_depths = [8, 3, 1]          # 256, 8, and 2 gray levels
    fig, axes = plt.subplots(1, len(bit_depths), figsize=(11, 3.5))
    for ax, bits in zip(axes, bit_depths):
        levels = 2 ** bits
        q = quantize(original, levels)
        mse = np.mean((original - q) ** 2)
        ax.imshow(q, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{bits}-bit ({levels} levels)\nMSE={mse:.5f}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q7_quantization_effect.png', dpi=160)
    plt.close()
    print("Q7: saved q7_quantization_effect.png "
          "(fewer quantization levels -> higher MSE -> visible false "
          "contouring / banding and loss of subtle intensity detail)")


if __name__ == "__main__":
    q7_quantization_effect()
