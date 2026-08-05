"""
Computer Vision - AT2 Concept Mapping
Question 3: Create a detailed concept map that accurately identifies
concepts related to image resolution and intensity levels, establishes
clear cause-effect relationships with image quality and usability,
integrates supporting evidence from literature, and presents the
concepts in a logically organized and visually clear format.

Demo: independently vary (a) spatial resolution (pixel grid size) and
(b) intensity/gray-level resolution (bits per pixel) on the same image,
to show how each dimension separately affects perceived image quality.

Run:  python3 cv_co1_at2_q3.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q3_resolution_and_intensity_levels.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def quantize(img, levels):
    step = 256 // levels
    q = (img // step) * step
    return q.astype(np.uint8)


def q3_resolution_and_intensity_levels():
    size = 256
    base = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(base, (128, 128), 90, 220, -1)
    cv2.circle(base, (128, 128), 60, 120, -1)
    cv2.circle(base, (128, 128), 30, 40, -1)

    spatial_res = [256, 64, 16]
    intensity_levels = [256, 8, 2]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    # Row 1: vary spatial resolution, keep full intensity levels
    for ax, res in zip(axes[0], spatial_res):
        small = cv2.resize(base, (res, res), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        ax.imshow(back, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"Spatial res: {res}x{res}\n(256 intensity levels)", fontsize=9)
        ax.axis('off')

    # Row 2: keep full spatial resolution, vary intensity levels
    for ax, levels in zip(axes[1], intensity_levels):
        q = quantize(base, levels)
        ax.imshow(q, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"Intensity levels: {levels}\n(256x256 spatial res)", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('q3_resolution_and_intensity_levels.png', dpi=160)
    plt.close()
    print("Q3: saved q3_resolution_and_intensity_levels.png "
          "(top row: lowering spatial resolution causes blockiness/loss of "
          "shape detail; bottom row: lowering intensity/gray-level "
          "resolution causes false contouring/banding, both reducing "
          "image quality and usability)")


if __name__ == "__main__":
    q3_resolution_and_intensity_levels()
