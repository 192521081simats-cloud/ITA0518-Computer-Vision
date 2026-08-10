"""
Experiment 19: Edge Detection under Illumination Changes
===========================================================
Objective:
    Study the effect of illumination variation (global brightness shifts,
    gamma / contrast changes, and non-uniform lighting) on edge-detection
    performance, and evaluate whether pre-processing (histogram
    equalization / CLAHE) improves robustness.

Tools:
    Python 3, OpenCV (cv2), NumPy, Matplotlib

Output:
    outputs/01_illumination_variants.png   - test image under different lighting
    outputs/02_canny_under_illumination.png- Canny edges for each illumination case
    outputs/03_clahe_recovery.png          - CLAHE-corrected edges vs raw
    outputs/04_metrics_plot.png            - edge similarity (F1 vs ground truth) per condition
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
np.random.seed(11)


# ---------------------------------------------------------------------------
# 1. Base scene + reference ("ground truth") edge map at nominal illumination
# ---------------------------------------------------------------------------
def make_scene(size=380):
    img = np.full((size, size), 100, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (170, 170), 200, -1)
    cv2.circle(img, (280, 100), 65, 60, -1)
    cv2.rectangle(img, (60, 220), (330, 250), 170, -1)
    pts = np.array([[80, 300], [240, 300], [160, 370]], np.int32)
    cv2.fillPoly(img, [pts], 220)
    return img


base = make_scene()
# add a touch of sensor noise so results are representative of real cameras
_noise = np.random.normal(0, 4, base.shape)
base = np.clip(base.astype(np.float32) + _noise, 0, 255).astype(np.uint8)
ground_truth_edges = cv2.Canny(base, 60, 150)


def apply_illumination(img, kind):
    f = img.astype(np.float32)
    if kind == "nominal":
        out = f
    elif kind == "dim (-60)":
        out = f - 60
    elif kind == "bright (+60)":
        out = f + 60
    elif kind == "low_contrast (gamma 0.4)":
        # compress the dynamic range into a narrow mid-gray band -> low
        # gradient magnitude at edges, which a FIXED Canny threshold misses
        out = 128 + (f - 128) * 0.18
    elif kind == "high_contrast (gamma 2.2)":
        out = 255 * (f / 255) ** 2.2
    elif kind == "uneven (vignette)":
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = 1 - 0.7 * (dist / dist.max())
        out = f * mask
    else:
        out = f
    return np.clip(out, 0, 255).astype(np.uint8)


conditions = ["nominal", "dim (-60)", "bright (+60)",
              "low_contrast (gamma 0.4)", "high_contrast (gamma 2.2)", "uneven (vignette)"]
variants = {c: apply_illumination(base, c) for c in conditions}

fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 4))
for ax, c in zip(axes, conditions):
    ax.imshow(variants[c], cmap="gray", vmin=0, vmax=255)
    ax.set_title(c, fontsize=10); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_illumination_variants.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 2. Canny edges with a FIXED threshold under each illumination condition
# ---------------------------------------------------------------------------
def edge_f1(pred, gt, tol=2):
    """F1 score between predicted and ground-truth edge maps allowing a
    small tolerance (dilation) for pixel-level misalignment."""
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gt_dil = cv2.dilate(gt, kernel)
    pred_dil = cv2.dilate(pred, kernel)
    tp = np.logical_and(pred > 0, gt_dil > 0).sum()
    fp = np.logical_and(pred > 0, gt_dil == 0).sum()
    fn = np.logical_and(gt > 0, pred_dil == 0).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


rows = []
fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 4))
for ax, c in zip(axes, conditions):
    edges = cv2.Canny(variants[c], 60, 150)  # fixed threshold for all
    p, r, f1 = edge_f1(edges, ground_truth_edges)
    rows.append({"condition": c, "stage": "raw", "precision": round(p, 3),
                 "recall": round(r, 3), "f1": round(f1, 3)})
    ax.imshow(edges, cmap="gray"); ax.set_title(f"{c}\nF1={f1:.2f}", fontsize=9); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_canny_under_illumination.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3. CLAHE pre-processing to recover robustness
# ---------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

fig, axes = plt.subplots(2, len(conditions), figsize=(4 * len(conditions), 8))
for col, c in enumerate(conditions):
    corrected = clahe.apply(variants[c])
    edges_corr = cv2.Canny(corrected, 60, 150)
    p, r, f1 = edge_f1(edges_corr, ground_truth_edges)
    rows.append({"condition": c, "stage": "CLAHE-corrected", "precision": round(p, 3),
                 "recall": round(r, 3), "f1": round(f1, 3)})
    axes[0, col].imshow(corrected, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"{c}\n(CLAHE)", fontsize=9); axes[0, col].axis("off")
    axes[1, col].imshow(edges_corr, cmap="gray")
    axes[1, col].set_title(f"F1={f1:.2f}", fontsize=9); axes[1, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_clahe_recovery.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 4. Metrics table + comparison plot (raw vs CLAHE-corrected F1)
# ---------------------------------------------------------------------------
import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["condition", "stage", "precision", "recall", "f1"])
    writer.writeheader()
    writer.writerows(rows)

raw_f1 = [r["f1"] for r in rows if r["stage"] == "raw"]
clahe_f1 = [r["f1"] for r in rows if r["stage"] == "CLAHE-corrected"]

x = np.arange(len(conditions))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width / 2, raw_f1, width, label="Raw (fixed threshold)", color="steelblue")
ax.bar(x + width / 2, clahe_f1, width, label="CLAHE-corrected", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=20, ha="right")
ax.set_ylabel("Edge-detection F1 score (vs. nominal ground truth)")
ax.set_title("Edge detection robustness across illumination conditions")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_metrics_plot.png"), dpi=150)
plt.close()

print("Done. Results written to", OUT)
for r in rows:
    print(r)
