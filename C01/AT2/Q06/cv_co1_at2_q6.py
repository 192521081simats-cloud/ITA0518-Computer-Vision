"""
Computer Vision - AT2 Concept Mapping
Question 6: Construct a concept map that identifies the complete workflow
of digital image formation, clearly linking sensing, sampling, and
quantization stages with their impact on image quality, integrating
relevant literature insights, and presenting the relationships in a
structured and visually coherent manner.

Demo: model image formation as f(x,y) = i(x,y) * r(x,y), where i is
illumination and r is scene reflectance, then take that continuous
irradiance function through sensing, sampling, and quantization stages.

Run:  python3 cv_co1_at2_q6.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q6_digital_image_formation_workflow.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q6_digital_image_formation_workflow():
    size = 256
    yy, xx = np.mgrid[0:size, 0:size]

    # Illumination component i(x,y): a soft light source gradient
    illumination = 0.5 + 0.5 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 120 ** 2))

    # Reflectance component r(x,y): scene surface reflectance pattern
    reflectance = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(reflectance, (60, 140), (200, 220), 0.9, -1)
    cv2.circle(reflectance, (128, 100), 40, 0.5, -1)
    reflectance += 0.05

    # Stage 1: Sensing - image formation f(x,y) = i(x,y) * r(x,y)
    sensed = illumination * reflectance
    sensed = np.clip(sensed, 0, 1)

    # Stage 2: Sampling - discretize spatial coordinates
    sample_res = 40
    small = cv2.resize(sensed, (sample_res, sample_res), interpolation=cv2.INTER_AREA)
    sampled = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)

    # Stage 3: Quantization - discretize amplitude into gray levels
    levels = 8
    quantized = np.floor(sampled * (levels - 1) + 0.5) / (levels - 1)

    stages = [
        ("Sensing\nf(x,y) = i(x,y)*r(x,y)", sensed),
        ("Sampling\n(spatial discretization)", sampled),
        ("Quantization\n(amplitude discretization)", quantized),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q6_digital_image_formation_workflow.png', dpi=160)
    plt.close()
    print("Q6: saved q6_digital_image_formation_workflow.png "
          "(sensing captures the product of illumination and reflectance "
          "as a continuous signal; sampling discretizes it spatially; "
          "quantization discretizes it in amplitude - each stage trades "
          "off fidelity for a representable digital image)")


if __name__ == "__main__":
    q6_digital_image_formation_workflow()
