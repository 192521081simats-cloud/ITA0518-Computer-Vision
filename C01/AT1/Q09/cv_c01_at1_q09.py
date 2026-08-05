"""
Computer Vision - AT1 Short Answer Test
Question 9: In an application like facial recognition, explain how
different levels of Computer Vision are involved from image acquisition
to decision making.

Demo: build a synthetic face-like scene and run it through
low-level (preprocessing) -> mid-level (feature extraction) ->
high-level (decision making) stages.

Run:  python3 q9.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q9_levels_of_cv_facial_recognition.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q9_levels_of_cv_facial_recognition():
    size = 300
    scene = np.full((size, size), 40, dtype=np.uint8)  # dim background
    # draw a simple synthetic "face"
    cv2.circle(scene, (150, 150), 90, 200, -1)          # face
    cv2.circle(scene, (115, 120), 12, 40, -1)            # left eye
    cv2.circle(scene, (185, 120), 12, 40, -1)            # right eye
    cv2.ellipse(scene, (150, 190), (35, 15), 0, 0, 180, 60, 4)  # mouth
    rng = np.random.default_rng(1)
    scene = np.clip(scene.astype(np.int16) + rng.normal(0, 6, scene.shape), 0, 255).astype(np.uint8)

    # --- Low-level vision: acquisition + pre-processing ---
    denoised = cv2.bilateralFilter(scene, 7, 50, 50)     # edge-preserving denoise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)                    # mild local contrast enhancement

    # --- Mid-level vision: feature/segment extraction ---
    edges = cv2.Canny(equalized, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    features_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(features_img, contours, -1, (0, 255, 0), 2)

    # --- High-level vision: decision making ---
    # simple rule-based "recognition": count prominent facial-region contours
    num_features = sum(1 for c in contours if cv2.contourArea(c) > 50)
    decision = "Face verified: candidate matches expected feature count" \
        if num_features >= 3 else "No reliable match found"

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    imgs = [scene, equalized, edges, cv2.cvtColor(features_img, cv2.COLOR_BGR2RGB)]
    titles = ["Acquisition\n(raw capture)",
              "Low-level\n(denoise + contrast)",
              "Mid-level\n(edge/feature extraction)",
              f"High-level (decision)\n{decision}"]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q9_levels_of_cv_facial_recognition.png', dpi=160)
    plt.close()
    print(f"Q9: saved q9_levels_of_cv_facial_recognition.png -> {decision} "
          f"(detected {num_features} candidate facial features)")


if __name__ == "__main__":
    q9_levels_of_cv_facial_recognition()
