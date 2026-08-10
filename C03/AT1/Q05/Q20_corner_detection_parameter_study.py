"""
Experiment 20: Corner Detection Parameter Study (Harris)
============================================================
Objective:
    Perform a systematic parameter-sensitivity analysis of the Harris
    corner detector: block size (neighbourhood window), Sobel aperture
    (ksize) and the empirical sensitivity constant k.

Tools:
    Python 3, OpenCV (cv2) cv2.cornerHarris, NumPy, Matplotlib

Output:
    outputs/01_test_image.png            - synthetic scene with known corners
    outputs/02_blocksize_grid.png        - effect of blockSize
    outputs/03_ksize_grid.png            - effect of Sobel aperture (ksize)
    outputs/04_k_grid.png                - effect of sensitivity constant k
    outputs/05_metrics_plot.png          - corner count vs each parameter
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
np.random.seed(5)


# ---------------------------------------------------------------------------
# 1. Test scene: chessboard-like pattern + polygons -> many well-defined corners
# ---------------------------------------------------------------------------
def make_scene(size=400):
    img = np.full((size, size), 50, dtype=np.uint8)
    step = 40
    for i, y in enumerate(range(0, size, step)):
        for j, x in enumerate(range(0, size, step)):
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + step, y + step), 220, -1)
    cv2.rectangle(img, (150, 150), (260, 260), 130, -1)
    pts = np.array([[300, 40], [370, 90], [340, 160]], np.int32)
    cv2.fillPoly(img, [pts], 90)
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


scene = make_scene()
plt.figure(figsize=(5, 5))
plt.imshow(scene, cmap="gray"); plt.title("Test scene (chessboard + shapes)"); plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_test_image.png"), dpi=150)
plt.close()

gray = np.float32(scene)


def harris_corners(gray_img, block_size, ksize, k, thresh_ratio=0.02):
    dst = cv2.cornerHarris(gray_img, block_size, ksize, k)
    dst_dilated = cv2.dilate(dst, None)
    thresh = thresh_ratio * dst_dilated.max()
    mask = dst_dilated > thresh
    ys, xs = np.where(mask)
    return mask, len(xs), dst


def overlay_corners(base_img, mask):
    vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    vis[mask] = [0, 0, 255]
    return vis


rows = []

# ---------------------------------------------------------------------------
# 2. Vary blockSize (neighbourhood size), fix ksize=3, k=0.04
# ---------------------------------------------------------------------------
block_sizes = [2, 4, 6, 10, 16]
fig, axes = plt.subplots(1, len(block_sizes), figsize=(4 * len(block_sizes), 4.2))
for ax, bs in zip(axes, block_sizes):
    mask, n, _ = harris_corners(gray, bs, 3, 0.04)
    vis = overlay_corners(scene, mask)
    rows.append({"study": "blockSize", "blockSize": bs, "ksize": 3, "k": 0.04, "num_corners": n})
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"blockSize={bs}\n{n} corners", fontsize=10); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_blocksize_grid.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Vary ksize (Sobel aperture), fix blockSize=4, k=0.04
# ---------------------------------------------------------------------------
ksizes = [3, 5, 7, 9, 11]
fig, axes = plt.subplots(1, len(ksizes), figsize=(4 * len(ksizes), 4.2))
for ax, ks in zip(axes, ksizes):
    mask, n, _ = harris_corners(gray, 4, ks, 0.04)
    vis = overlay_corners(scene, mask)
    rows.append({"study": "ksize", "blockSize": 4, "ksize": ks, "k": 0.04, "num_corners": n})
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"ksize={ks}\n{n} corners", fontsize=10); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_ksize_grid.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Vary k (sensitivity constant), fix blockSize=4, ksize=3
# ---------------------------------------------------------------------------
ks_vals = [0.02, 0.04, 0.06, 0.10, 0.15]
fig, axes = plt.subplots(1, len(ks_vals), figsize=(4 * len(ks_vals), 4.2))
for ax, kval in zip(axes, ks_vals):
    mask, n, _ = harris_corners(gray, 4, 3, kval)
    vis = overlay_corners(scene, mask)
    rows.append({"study": "k", "blockSize": 4, "ksize": 3, "k": kval, "num_corners": n})
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"k={kval}\n{n} corners", fontsize=10); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_k_grid.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 5. Metrics table + trend plots
# ---------------------------------------------------------------------------
import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["study", "blockSize", "ksize", "k", "num_corners"])
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
bs_rows = [r for r in rows if r["study"] == "blockSize"]
axes[0].plot([r["blockSize"] for r in bs_rows], [r["num_corners"] for r in bs_rows], "o-", color="teal")
axes[0].set_xlabel("blockSize"); axes[0].set_ylabel("Detected corners")
axes[0].set_title("Corners vs. blockSize"); axes[0].grid(alpha=0.3)

ks_rows = [r for r in rows if r["study"] == "ksize"]
axes[1].plot([r["ksize"] for r in ks_rows], [r["num_corners"] for r in ks_rows], "s-", color="darkorange")
axes[1].set_xlabel("ksize (Sobel aperture)"); axes[1].set_ylabel("Detected corners")
axes[1].set_title("Corners vs. ksize"); axes[1].grid(alpha=0.3)

k_rows = [r for r in rows if r["study"] == "k"]
axes[2].plot([r["k"] for r in k_rows], [r["num_corners"] for r in k_rows], "^-", color="crimson")
axes[2].set_xlabel("k (sensitivity constant)"); axes[2].set_ylabel("Detected corners")
axes[2].set_title("Corners vs. k"); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_metrics_plot.png"), dpi=150)
plt.close()

print("Done. Results written to", OUT)
for r in rows:
    print(r)
