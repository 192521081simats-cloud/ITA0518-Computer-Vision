"""
Experiment 16: Threshold Effect in Edge Detection
===================================================
Objective:
    Study how the choice of threshold values (used inside the Canny edge
    detector, and as a simple magnitude threshold applied to the Sobel
    gradient) influences the quality of the detected edges: edge
    continuity, accuracy and sensitivity to noise.

Tools:
    Python 3, OpenCV (cv2), NumPy, Matplotlib

Output:
    outputs/01_test_image.png            - synthetic test image + noisy version
    outputs/02_canny_threshold_grid.png  - Canny results for several threshold pairs
    outputs/03_sobel_threshold_grid.png  - simple gradient-threshold results
    outputs/04_metrics_plot.png          - edge-count / continuity vs threshold
    outputs/metrics.csv                  - numeric results table
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Build a synthetic test scene (known geometry -> "ground truth" edges)
# ---------------------------------------------------------------------------
def make_test_image(size=400):
    img = np.full((size, size), 40, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (180, 180), 200, -1)
    cv2.circle(img, (290, 110), 70, 120, -1)
    cv2.rectangle(img, (60, 230), (340, 260), 160, -1)
    pts = np.array([[80, 300], [250, 300], [165, 380]], np.int32)
    cv2.fillPoly(img, [pts], 90)
    # smooth gradient band to test low-contrast edges
    for x in range(200, 400):
        img[300:340, x] = int(60 + (x - 200) * (150 / 200))
    return img


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


clean = make_test_image()
noisy = add_gaussian_noise(clean, sigma=18)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(clean, cmap="gray"); ax[0].set_title("Clean synthetic image"); ax[0].axis("off")
ax[1].imshow(noisy, cmap="gray"); ax[1].set_title("With Gaussian noise (sigma=18)"); ax[1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_test_image.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 2. Canny edge detection across several (low, high) threshold pairs
# ---------------------------------------------------------------------------
threshold_pairs = [(10, 30), (30, 90), (60, 150), (100, 200), (150, 250)]

def edge_continuity_score(edge_map):
    """Fraction of edge pixels that belong to connected components of
    length >= 15 px (a proxy for 'continuous' vs 'broken/fragmented' edges)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
    if n_labels <= 1:
        return 0.0
    sizes = stats[1:, cv2.CC_STAT_AREA]
    long_pixels = sizes[sizes >= 15].sum()
    total_pixels = sizes.sum()
    return 0.0 if total_pixels == 0 else long_pixels / total_pixels


rows = []
fig, axes = plt.subplots(2, len(threshold_pairs), figsize=(4 * len(threshold_pairs), 8))
for i, src_name, src in [(0, "clean", clean), (1, "noisy", noisy)]:
    pass

for col, (lo, hi) in enumerate(threshold_pairs):
    for row, (name, src) in enumerate([("clean", clean), ("noisy", noisy)]):
        edges = cv2.Canny(src, lo, hi)
        density = edges.mean() / 255.0
        continuity = edge_continuity_score(edges)
        rows.append({"image": name, "low": lo, "high": hi,
                      "edge_density_%": round(density * 100, 3),
                      "continuity_score": round(continuity, 3)})
        axes[row, col].imshow(edges, cmap="gray")
        axes[row, col].set_title(f"{name}\nT=({lo},{hi})", fontsize=10)
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_canny_threshold_grid.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3. Simple gradient-magnitude thresholding (Sobel) for comparison
# ---------------------------------------------------------------------------
def sobel_magnitude(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return (255 * mag / mag.max()).astype(np.uint8)

sobel_thresholds = [20, 40, 80, 120, 180]
mag_clean = sobel_magnitude(clean)
mag_noisy = sobel_magnitude(noisy)

fig, axes = plt.subplots(2, len(sobel_thresholds), figsize=(4 * len(sobel_thresholds), 8))
for col, t in enumerate(sobel_thresholds):
    for row, (name, mag) in enumerate([("clean", mag_clean), ("noisy", mag_noisy)]):
        edges = (mag > t).astype(np.uint8) * 255
        density = edges.mean() / 255.0
        continuity = edge_continuity_score(edges)
        rows.append({"image": f"{name}_sobel", "low": "-", "high": t,
                      "edge_density_%": round(density * 100, 3),
                      "continuity_score": round(continuity, 3)})
        axes[row, col].imshow(edges, cmap="gray")
        axes[row, col].set_title(f"{name} sobel\nT={t}", fontsize=10)
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_sobel_threshold_grid.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 4. Summarize numerically and plot trends
# ---------------------------------------------------------------------------
import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

canny_clean = [r for r in rows if r["image"] == "clean"]
canny_noisy = [r for r in rows if r["image"] == "noisy"]
highs = [r["high"] for r in canny_clean]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(highs, [r["edge_density_%"] for r in canny_clean], "o-", label="clean")
ax[0].plot(highs, [r["edge_density_%"] for r in canny_noisy], "s-", label="noisy")
ax[0].set_xlabel("High threshold"); ax[0].set_ylabel("Edge pixel density (%)")
ax[0].set_title("Edge density vs. threshold"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(highs, [r["continuity_score"] for r in canny_clean], "o-", label="clean")
ax[1].plot(highs, [r["continuity_score"] for r in canny_noisy], "s-", label="noisy")
ax[1].set_xlabel("High threshold"); ax[1].set_ylabel("Continuity score")
ax[1].set_title("Edge continuity vs. threshold"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_metrics_plot.png"), dpi=150)
plt.close()

print("Done. Results written to", OUT)
for r in rows:
    print(r)
