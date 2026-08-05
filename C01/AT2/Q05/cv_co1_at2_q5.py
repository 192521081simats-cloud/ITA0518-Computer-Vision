"""
Computer Vision - AT2 Concept Mapping
Question 5: Construct an original and logically structured concept map
that identifies system capabilities and their influence on the selection
and effectiveness of vision-based applications, establishes clear
interconnections, integrates multiple literature perspectives, and
maintains high clarity and visual organization.

Demo: simulate a moving object captured by systems with different
capabilities (frame rate / exposure time) and show that only a system
with a high enough frame rate and short enough exposure is effective for
a motion-sensitive application (e.g. tracking a fast-moving ball).

Run:  python3 cv_co1_at2_q5.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q5_system_capabilities_vs_application_effectiveness.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def render_ball_with_exposure(size, position, speed_px, exposure_smear):
    """Render a moving ball; exposure_smear approximates motion blur caused
    by a longer exposure time relative to the object's speed."""
    frame = np.zeros((size, size), dtype=np.uint8)
    x, y = position
    n_ghosts = max(1, exposure_smear)
    for i in range(n_ghosts):
        cv2.circle(frame, (int(x - i * speed_px / n_ghosts), y), 12,
                   int(180 / n_ghosts), -1)
    return frame


def q5_system_capabilities_vs_application_effectiveness():
    size = 256
    speed_px = 40  # object displacement during one exposure, in pixels

    systems = [
        ("Low-end system\n(slow frame rate,\nlong exposure)", 20),
        ("Mid-range system\n(moderate exposure)", 6),
        ("High-performance system\n(high frame rate,\nshort exposure)", 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (title, smear) in zip(axes, systems):
        frame = render_ball_with_exposure(size, (180, 128), speed_px, smear)
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trackable = len(contours) > 0 and cv2.contourArea(max(contours, key=cv2.contourArea)) < 900
        ax.imshow(frame, cmap='gray')
        ax.set_title(f"{title}\nball position trackable: {trackable}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('q5_system_capabilities_vs_application_effectiveness.png', dpi=160)
    plt.close()
    print("Q5: saved q5_system_capabilities_vs_application_effectiveness.png "
          "(a system's capabilities - here frame rate and exposure time - "
          "determine whether it can effectively support a given "
          "vision-based application, such as tracking a fast-moving "
          "object; smearing/ghosting makes precise tracking unreliable)")


if __name__ == "__main__":
    q5_system_capabilities_vs_application_effectiveness()
