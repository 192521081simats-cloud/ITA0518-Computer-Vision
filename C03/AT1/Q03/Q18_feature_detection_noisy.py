"""
Experiment 18: Feature Detection in Noisy Images
==================================================
Objective:
    Analyze how increasing image noise affects keypoint/feature detection
    (using ORB, a fast binary-descriptor detector) in terms of the number
    of keypoints found, their stability (repeatability against the
    clean/reference image), and matching accuracy.

Tools:
    Python 3, OpenCV (cv2) ORB detector + BFMatcher, NumPy, Matplotlib

Output:
    outputs/01_test_scene.png              - reference textured scene
    outputs/02_keypoints_by_noise.png      - keypoints overlaid at each noise level
    outputs/03_matches_examples.png        - feature matches: clean vs noisy
    outputs/04_metrics_plot.png            - keypoint count & good-match ratio vs noise
    outputs/metrics.csv
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
np.random.seed(3)


# ---------------------------------------------------------------------------
# 1. Build a richly textured synthetic scene (features need texture/corners)
# ---------------------------------------------------------------------------
def make_textured_scene(size=400):
    img = np.full((size, size), 60, dtype=np.uint8)
    rng = np.random.RandomState(1)
    for _ in range(40):
        x, y = rng.randint(20, size - 20, 2)
        r = rng.randint(8, 26)
        val = int(rng.randint(90, 230))
        shape = rng.randint(0, 3)
        if shape == 0:
            cv2.circle(img, (x, y), r, val, -1)
        elif shape == 1:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), val, -1)
        else:
            pts = np.array([[x, y - r], [x - r, y + r], [x + r, y + r]], np.int32)
            cv2.fillPoly(img, [pts], val)
    checker = np.indices((size, size)).sum(axis=0) % 40 < 2
    img[checker] = np.clip(img[checker].astype(int) + 40, 0, 255).astype(np.uint8)
    return img


clean = make_textured_scene()

plt.figure(figsize=(5, 5))
plt.imshow(clean, cmap="gray"); plt.title("Reference textured scene"); plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_test_scene.png"), dpi=150)
plt.close()


def add_gaussian_noise(img, sigma):
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


noise_levels = [0, 10, 20, 30, 45, 60]
noisy_images = {s: (clean if s == 0 else add_gaussian_noise(clean, s)) for s in noise_levels}


# ---------------------------------------------------------------------------
# 2. ORB feature detection at each noise level
# ---------------------------------------------------------------------------
orb = cv2.ORB_create(nfeatures=500)

kp_ref, des_ref = orb.detectAndCompute(clean, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

results = []
fig, axes = plt.subplots(1, len(noise_levels), figsize=(4 * len(noise_levels), 4.2))
for ax, sigma in zip(axes, noise_levels):
    img = noisy_images[sigma]
    kp, des = orb.detectAndCompute(img, None)
    vis = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(vis); ax.set_title(f"sigma={sigma}\n{len(kp)} keypoints", fontsize=10); ax.axis("off")

    # Match against reference to assess stability / repeatability
    if des is not None and des_ref is not None and len(des) > 0:
        matches = bf.knnMatch(des_ref, des, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        match_ratio = len(good) / max(1, len(kp_ref))
    else:
        good, match_ratio = [], 0.0

    results.append({"noise_sigma": sigma, "num_keypoints": len(kp),
                     "good_matches_vs_reference": len(good),
                     "match_ratio": round(match_ratio, 3)})
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_keypoints_by_noise.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3. Visualize explicit feature matches: clean vs. two noise levels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
for ax, sigma in zip(axes, [20, 60]):
    img = noisy_images[sigma]
    kp, des = orb.detectAndCompute(img, None)
    matches = bf.knnMatch(des_ref, des, k=2) if des is not None else []
    good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance]
    good = sorted(good, key=lambda m: m.distance)[:40]
    match_img = cv2.drawMatches(clean, kp_ref, img, kp, good, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax.imshow(match_img); ax.set_title(f"Reference vs. noisy (sigma={sigma}): {len(good)} good matches shown")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_matches_examples.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 4. Metrics table + trend plot
# ---------------------------------------------------------------------------
import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

sigmas = [r["noise_sigma"] for r in results]
kpc = [r["num_keypoints"] for r in results]
ratio = [r["match_ratio"] for r in results]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sigmas, kpc, "o-", color="darkorange")
ax[0].set_xlabel("Noise sigma"); ax[0].set_ylabel("Number of ORB keypoints")
ax[0].set_title("Keypoint count vs. noise level"); ax[0].grid(alpha=0.3)

ax[1].plot(sigmas, ratio, "s-", color="crimson")
ax[1].set_xlabel("Noise sigma"); ax[1].set_ylabel("Good-match ratio (vs. reference)")
ax[1].set_title("Feature stability / repeatability vs. noise"); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_metrics_plot.png"), dpi=150)
plt.close()

print("Done. Results written to", OUT)
for r in results:
    print(r)
