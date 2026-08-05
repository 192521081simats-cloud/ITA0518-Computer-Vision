"""
Computer Vision - AT2 Concept Mapping
Question 4: Design an analytically organized concept map that identifies
key factors in image acquisition conditions, clearly links them to
downstream processing and application performance through meaningful
relationships, integrates findings from relevant studies, and ensures a
well-structured and easy-to-interpret presentation.

Demo: simulate three acquisition conditions (ideal, low light + sensor
noise, motion blur) on the same scene, then run identical downstream
processing (edge detection) on each and compare how many true object
edges are actually recovered.

Run:  python3 cv_co1_at2_q4.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q4_acquisition_conditions_vs_performance.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_scene():
    size = 256
    scene = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(scene, (60, 60), (200, 200), 200, -1)
    cv2.circle(scene, (128, 128), 40, 60, -1)
    return scene


def q4_acquisition_conditions_vs_performance():
    scene = make_scene()

    # Condition A: ideal acquisition
    ideal = scene.copy()

    # Condition B: low light + sensor noise
    low_light = (scene.astype(np.float32) * 0.35).astype(np.uint8)
    noise = np.random.normal(0, 25, low_light.shape).astype(np.int16)
    low_light = np.clip(low_light.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Condition C: motion blur during exposure
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    motion_blur = cv2.filter2D(scene, -1, kernel)

    conditions = [
        ("Ideal acquisition", ideal),
        ("Low light + noise", low_light),
        ("Motion blur", motion_blur),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for col, (title, img) in enumerate(conditions):
        edges = cv2.Canny(img, 50, 150)
        edge_count = int(np.sum(edges > 0))

        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(edges, cmap='gray')
        axes[1, col].set_title(f"Edges detected: {edge_count} px", fontsize=9)
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig('q4_acquisition_conditions_vs_performance.png', dpi=160)
    plt.close()
    print("Q4: saved q4_acquisition_conditions_vs_performance.png "
          "(poor acquisition conditions - low light/noise or motion blur - "
          "degrade the edge map that identical downstream processing can "
          "recover, directly lowering the performance of any application "
          "built on top of it)")


if __name__ == "__main__":
    q4_acquisition_conditions_vs_performance()
