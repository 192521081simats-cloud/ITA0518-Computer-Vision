"""
Computer Vision - AT1 Short Answer Test
Question 10: An image captured with low resolution fails to detect
objects clearly. Explain how sampling and image formation affect object
visibility.

Demo: downsample (simulate low-resolution capture) then measure how well
a small object can still be detected as sampling grid gets coarser.

Run:  python3 q10.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q10_sampling_and_object_visibility.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q10_sampling_and_object_visibility():
    size = 256
    scene = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(scene, (128, 128), 60, 255, -1)   # large object: a car body
    cv2.circle(scene, (190, 90), 6, 255, -1)     # small object: a pedestrian, far away

    resolutions = [size, 64, 24]   # simulate progressively coarser sampling
    fig, axes = plt.subplots(1, len(resolutions), figsize=(11, 3.8))
    for ax, res in zip(axes, resolutions):
        small = cv2.resize(scene, (res, res), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        _, thresh = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_small_object = any(5 < cv2.contourArea(c) < 400 for c in contours)
        ax.imshow(back, cmap='gray')
        ax.set_title(f"Sampled at {res}x{res}\nsmall object detected: {detected_small_object}",
                     fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q10_sampling_and_object_visibility.png', dpi=160)
    plt.close()
    print("Q10: saved q10_sampling_and_object_visibility.png "
          "(as spatial sampling coarsens, the small/distant object falls "
          "below the sampling grid and disappears, while the large object "
          "is still visible but loses shape detail)")


if __name__ == "__main__":
    q10_sampling_and_object_visibility()
