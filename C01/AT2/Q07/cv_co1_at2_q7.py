"""
Computer Vision - AT2 Concept Mapping
Question 7: Develop a concept map that identifies different levels of
Computer Vision, establishes relationships between low-level processing,
mid-level analysis, and high-level interpretation, integrates
literature-based examples, and ensures clear hierarchical organization.

Demo: process one synthetic scene through the three classical levels of
computer vision:
    low-level (edge/gradient extraction)
    mid-level (segmentation / grouping into regions or blobs)
    high-level (interpretation: counting and labeling recognized objects)

Run:  python3 cv_co1_at2_q7.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q7_levels_of_computer_vision.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q7_levels_of_computer_vision():
    size = 256
    scene = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(scene, (80, 80), 40, 180, -1)
    cv2.rectangle(scene, (150, 60), (220, 130), 220, -1)
    cv2.circle(scene, (150, 190), 35, 150, -1)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)

    # Low-level: gradient/edge extraction (pixel-level operations)
    low_level = cv2.Canny(scene, 50, 150)

    # Mid-level: segmentation into meaningful regions/blobs
    _, thresh = cv2.threshold(scene, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mid_level = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, c in enumerate(contours):
        cv2.drawContours(mid_level, [c], -1, colors[i % len(colors)], 3)

    # High-level: interpretation - classify each region and label it
    high_level = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    labels = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h else 0
        shape = "square-ish object" if 0.85 < aspect < 1.15 and area > 3000 else "round object"
        labels.append((x, y, shape))
        cv2.rectangle(high_level, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(high_level, shape, (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    stages = [
        ("Low-level\n(edges/gradients)", low_level, 'gray'),
        ("Mid-level\n(segmented regions)", cv2.cvtColor(mid_level, cv2.COLOR_BGR2RGB), None),
        (f"High-level\n({len(labels)} objects interpreted)", cv2.cvtColor(high_level, cv2.COLOR_BGR2RGB), None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, img, cmap) in zip(axes, stages):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q7_levels_of_computer_vision.png', dpi=160)
    plt.close()
    print("Q7: saved q7_levels_of_computer_vision.png "
          "(low-level processing extracts raw pixel features such as "
          "edges, mid-level analysis groups these into coherent regions, "
          "and high-level interpretation assigns meaning/labels to those "
          "regions - each level depending on the one below it)")


if __name__ == "__main__":
    q7_levels_of_computer_vision()
