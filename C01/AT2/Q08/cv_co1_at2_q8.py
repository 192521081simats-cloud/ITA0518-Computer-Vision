"""
Computer Vision - AT2 Concept Mapping
Question 8: Create a concept map that identifies factors affecting
spatial resolution, clearly linking them to sampling rate, pixel
density, and application requirements, incorporating supporting evidence
from literature, and presenting the structure with clarity.

Demo: place two objects at a fixed physical separation and vary the
sensor's pixel density (equivalently, the sampling rate) to find the
minimum pixel density at which the two objects can still be resolved as
separate, illustrating a Nyquist-like resolving-power requirement.

Run:  python3 cv_co1_at2_q8.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q8_spatial_resolution_factors.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def two_dot_scene(size=256, separation=14, radius=5):
    scene = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2
    cv2.circle(scene, (cx - separation, size // 2), radius, 255, -1)
    cv2.circle(scene, (cx + separation, size // 2), radius, 255, -1)
    return scene


def q8_spatial_resolution_factors():
    size = 256
    scene = two_dot_scene(size)

    pixel_densities = [size, 32, 16, 10]  # simulated sensor sampling grids

    fig, axes = plt.subplots(1, len(pixel_densities), figsize=(13, 3.8))
    for ax, res in zip(axes, pixel_densities):
        small = cv2.resize(scene, (res, res), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        _, thresh = cv2.threshold(back, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        resolved = len(contours) >= 2
        ax.imshow(back, cmap='gray')
        ax.set_title(f"Pixel density: {res}x{res}\ntwo objects resolved: {resolved}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('q8_spatial_resolution_factors.png', dpi=160)
    plt.close()
    print("Q8: saved q8_spatial_resolution_factors.png "
          "(spatial resolution is governed by sampling rate/pixel "
          "density relative to object separation; once pixel density "
          "falls below what the application requires, distinct nearby "
          "objects merge into one and can no longer be resolved)")


if __name__ == "__main__":
    q8_spatial_resolution_factors()
