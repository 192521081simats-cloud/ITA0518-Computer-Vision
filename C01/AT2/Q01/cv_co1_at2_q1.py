"""
Computer Vision - AT2 Concept Mapping
Question 1: Construct a comprehensive and well-organized concept map that
accurately identifies all key concepts from literature involved in the
pipeline from real-world scene capture to digital image representation,
establishes clear and meaningful relationships among intermediate
transformations and dependencies, integrates insights from multiple
studies, and presents the information in a clear and visually structured
manner.

Demo: simulate the full pipeline from a "real-world" continuous scene to
its final digital image representation:
    continuous scene -> optical blur (lens) -> spatial sampling (sensor
    grid) -> intensity quantization (ADC) -> digital image f(x,y)

Run:  python3 cv_co1_at2_q1.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q1_scene_to_digital_image_pipeline.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q1_scene_to_digital_image_pipeline():
    size = 512

    # 1. Real-world continuous scene (approximated at high resolution)
    scene = np.zeros((size, size), dtype=np.float32)
    cv2.circle(scene, (256, 256), 150, 1.0, -1)      # a "sun"
    cv2.rectangle(scene, (60, 380), (220, 460), 0.6, -1)  # a "building"
    yy, xx = np.mgrid[0:size, 0:size]
    scene += 0.15 * np.sin(xx / 9.0) * np.sin(yy / 9.0)   # fine texture/detail
    scene = np.clip(scene, 0, 1)

    # 2. Image formation: optical system introduces blur (lens/aperture)
    optical = cv2.GaussianBlur(scene, (0, 0), sigmaX=3)

    # 3. Spatial sampling: sensor captures on a discrete pixel grid
    sample_res = 48
    sampled_small = cv2.resize(optical, (sample_res, sample_res), interpolation=cv2.INTER_AREA)
    sampled = cv2.resize(sampled_small, (size, size), interpolation=cv2.INTER_NEAREST)

    # 4. Intensity quantization: ADC maps continuous irradiance to discrete levels
    levels = 4  # e.g. a coarse 2-bit-like quantizer for demonstration
    quantized = np.floor(sampled * (levels - 1) + 0.5) / (levels - 1)

    stages = [
        ("1. Real-world scene\n(continuous)", scene),
        ("2. Optical image\n(lens blur)", optical),
        ("3. Sampled image\n(sensor grid)", sampled),
        ("4. Digital image f(x,y)\n(quantized)", quantized),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q1_scene_to_digital_image_pipeline.png', dpi=160)
    plt.close()
    print("Q1: saved q1_scene_to_digital_image_pipeline.png "
          "(pipeline: continuous real-world scene -> optical blur from the "
          "lens -> spatial sampling on the sensor grid -> intensity "
          "quantization -> final digital image representation)")


if __name__ == "__main__":
    q1_scene_to_digital_image_pipeline()
