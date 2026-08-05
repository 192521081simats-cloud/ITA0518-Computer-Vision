"""
Computer Vision - AT2 Concept Mapping
Question 2: Develop a hierarchically structured concept map that
identifies major stages of visual data handling in a Computer Vision
system, clearly illustrates logical relationships and contributions of
each stage toward generating meaningful outputs, incorporates relevant
literature connections, and ensures clarity and readability.

Demo: run a synthetic image through the major stages of a CV system:
    acquisition -> preprocessing -> feature extraction -> analysis
    (interpretation) -> meaningful output (annotated result)

Run:  python3 cv_co1_at2_q2.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q2_stages_of_visual_data_handling.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q2_stages_of_visual_data_handling():
    size = 256

    # Stage 1: Acquisition - raw noisy image straight from a sensor
    acquired = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(acquired, (60, 60), (180, 180), 200, -1)
    cv2.circle(acquired, (190, 60), 30, 150, -1)
    noise = np.random.normal(0, 20, acquired.shape).astype(np.int16)
    acquired = np.clip(acquired.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Stage 2: Preprocessing - denoise / normalize
    preprocessed = cv2.GaussianBlur(acquired, (5, 5), 0)

    # Stage 3: Feature extraction - edges as low-level features
    features = cv2.Canny(preprocessed, 50, 150)

    # Stage 4: Analysis / interpretation - find and label shapes
    contours, _ = cv2.findContours(features, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analysis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(analysis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    stages = [
        ("1. Acquisition\n(raw sensor data)", acquired, 'gray'),
        ("2. Preprocessing\n(denoise/normalize)", preprocessed, 'gray'),
        ("3. Feature extraction\n(edges)", features, 'gray'),
        ("4. Analysis -> Output\n(detected objects)", cv2.cvtColor(analysis, cv2.COLOR_BGR2RGB), None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, img, cmap) in zip(axes, stages):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q2_stages_of_visual_data_handling.png', dpi=160)
    plt.close()
    print("Q2: saved q2_stages_of_visual_data_handling.png "
          "(each stage - acquisition, preprocessing, feature extraction, "
          "and analysis - builds on the previous one's output to produce "
          "a meaningful, interpretable final result)")


if __name__ == "__main__":
    q2_stages_of_visual_data_handling()
