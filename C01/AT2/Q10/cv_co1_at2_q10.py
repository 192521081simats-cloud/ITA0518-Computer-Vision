"""
Computer Vision - AT2 Concept Mapping
Question 10: Construct a concept map that identifies the relationship
between sampling, aliasing, and image distortion, clearly illustrating
cause-effect dependencies, integrating relevant theoretical insights, and
presenting the structure in a visually organized manner.

Demo: generate a high-frequency sinusoidal stripe pattern and sample it
at rates above and below the Nyquist rate. Sampling below the Nyquist
rate produces a moire pattern - a classic visible signature of aliasing
distortion caused purely by the sampling stage, not by the original
scene content.

Run:  python3 cv_co1_at2_q10.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q10_sampling_aliasing_and_distortion.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q10_sampling_aliasing_and_distortion():
    size = 512
    x = np.arange(size)
    freq = 0.45  # cycles per pixel in the original continuous-like pattern
    stripes = (0.5 + 0.5 * np.sin(2 * np.pi * freq * x))
    scene = np.tile(stripes, (size, 1))
    scene_u8 = (scene * 255).astype(np.uint8)

    # Sampling rates relative to signal frequency: above, near, and below Nyquist
    sample_res = [size, 40, 18]
    captions = [
        f"Sampled at {size}px\n(sampling rate >> 2x freq: faithful)",
        f"Sampled at 40px\n(near Nyquist limit: distortion starts)",
        f"Sampled at 18px\n(below Nyquist: aliasing / moire)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, res, caption in zip(axes, sample_res, captions):
        small = cv2.resize(scene_u8, (res, res), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        ax.imshow(back, cmap='gray')
        ax.set_title(caption, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('q10_sampling_aliasing_and_distortion.png', dpi=160)
    plt.close()
    print("Q10: saved q10_sampling_aliasing_and_distortion.png "
          "(when the sampling rate drops below twice the signal's "
          "frequency - the Nyquist rate - the true high-frequency stripe "
          "pattern can no longer be represented, and a false low-frequency "
          "moire pattern appears instead: a distortion introduced purely "
          "by under-sampling, not present in the original scene)")


if __name__ == "__main__":
    q10_sampling_aliasing_and_distortion()
