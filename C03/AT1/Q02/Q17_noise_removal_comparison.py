"""
Experiment 17: Noise Removal Comparison
========================================
Objective:
    Compare several classical noise-removal (denoising) filters and
    quantify how well each preserves image detail while removing noise.

Tools:
    Python 3, OpenCV (cv2), NumPy, Matplotlib, scikit-image (metrics)

Output:
    outputs/01_noisy_inputs.png     - clean image + gaussian/salt&pepper/speckle noise
    outputs/02_filters_gaussian.png - filter comparison for gaussian noise
    outputs/03_filters_sp.png       - filter comparison for salt & pepper noise
    outputs/04_metrics_bar.png      - PSNR / SSIM bar charts
    outputs/metrics.csv
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
np.random.seed(7)


# ---------------------------------------------------------------------------
# 1. Test image + three noise models
# ---------------------------------------------------------------------------
def make_test_image(size=384):
    img = np.zeros((size, size), np.uint8)
    cv2.rectangle(img, (30, 30), (170, 170), 210, -1)
    cv2.circle(img, (280, 100), 70, 130, -1)
    cv2.rectangle(img, (50, 220), (330, 250), 170, -1)
    pts = np.array([[70, 300], [230, 300], [150, 370]], np.int32)
    cv2.fillPoly(img, [pts], 90)
    x = np.linspace(0, 1, size)
    grad = (60 + 100 * x).astype(np.uint8)
    img[:, :] = np.maximum(img, np.tile(grad, (size, 1)) // 4)
    return img


def gaussian_noise(img, sigma=25):
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def salt_pepper_noise(img, amount=0.05):
    out = img.copy()
    n_pixels = img.size
    n_salt = int(amount * n_pixels * 0.5)
    n_pepper = int(amount * n_pixels * 0.5)
    ys = np.random.randint(0, img.shape[0], n_salt)
    xs = np.random.randint(0, img.shape[1], n_salt)
    out[ys, xs] = 255
    ys = np.random.randint(0, img.shape[0], n_pepper)
    xs = np.random.randint(0, img.shape[1], n_pepper)
    out[ys, xs] = 0
    return out


def speckle_noise(img, var=0.05):
    n = np.random.randn(*img.shape) * np.sqrt(var)
    out = img.astype(np.float32) + img.astype(np.float32) * n
    return np.clip(out, 0, 255).astype(np.uint8)


clean = make_test_image()
noisy_gauss = gaussian_noise(clean)
noisy_sp = salt_pepper_noise(clean)
noisy_speckle = speckle_noise(clean)

fig, ax = plt.subplots(1, 4, figsize=(14, 4))
for a, im, t in zip(ax, [clean, noisy_gauss, noisy_sp, noisy_speckle],
                     ["Clean", "Gaussian noise", "Salt & Pepper", "Speckle noise"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_noisy_inputs.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 2. Filters under test
# ---------------------------------------------------------------------------
def apply_filters(img):
    return {
        "Mean (5x5)": cv2.blur(img, (5, 5)),
        "Gaussian (5x5)": cv2.GaussianBlur(img, (5, 5), 1.2),
        "Median (5x5)": cv2.medianBlur(img, 5),
        "Bilateral": cv2.bilateralFilter(img, 9, 75, 75),
        "Non-Local Means": cv2.fastNlMeansDenoising(img, None, h=22, templateWindowSize=7, searchWindowSize=21),
    }


def evaluate(clean_img, noisy_img, filtered_dict):
    results = []
    base_psnr = psnr(clean_img, noisy_img)
    base_ssim = ssim(clean_img, noisy_img)
    results.append({"filter": "No filtering (noisy)", "psnr": base_psnr, "ssim": base_ssim})
    for name, out in filtered_dict.items():
        results.append({"filter": name, "psnr": psnr(clean_img, out), "ssim": ssim(clean_img, out)})
    return results


def plot_grid(clean_img, noisy_img, filtered_dict, title, fname):
    n = len(filtered_dict) + 2
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    axes[0].imshow(clean_img, cmap="gray"); axes[0].set_title("Clean"); axes[0].axis("off")
    axes[1].imshow(noisy_img, cmap="gray"); axes[1].set_title("Noisy"); axes[1].axis("off")
    for a, (name, out) in zip(axes[2:], filtered_dict.items()):
        a.imshow(out, cmap="gray"); a.set_title(name, fontsize=9); a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()


filt_gauss = apply_filters(noisy_gauss)
filt_sp = apply_filters(noisy_sp)
filt_speckle = apply_filters(noisy_speckle)

plot_grid(clean, noisy_gauss, filt_gauss, "Filters applied to Gaussian noise", "02_filters_gaussian.png")
plot_grid(clean, noisy_sp, filt_sp, "Filters applied to Salt & Pepper noise", "03_filters_sp.png")

# also save speckle grid for completeness
plot_grid(clean, noisy_speckle, filt_speckle, "Filters applied to Speckle noise", "03b_filters_speckle.png")


# ---------------------------------------------------------------------------
# 3. Metrics table + bar chart
# ---------------------------------------------------------------------------
all_rows = []
for noise_name, noisy_img, filt in [("gaussian", noisy_gauss, filt_gauss),
                                     ("salt_pepper", noisy_sp, filt_sp),
                                     ("speckle", noisy_speckle, filt_speckle)]:
    for r in evaluate(clean, noisy_img, filt):
        r["noise_type"] = noise_name
        all_rows.append(r)

import csv
with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["noise_type", "filter", "psnr", "ssim"])
    writer.writeheader()
    for r in all_rows:
        writer.writerow({"noise_type": r["noise_type"], "filter": r["filter"],
                          "psnr": round(r["psnr"], 2), "ssim": round(r["ssim"], 3)})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, noise_name in zip(axes, ["gaussian", "salt_pepper", "speckle"]):
    subset = [r for r in all_rows if r["noise_type"] == noise_name]
    names = [r["filter"] for r in subset]
    psnrs = [r["psnr"] for r in subset]
    ax.bar(range(len(names)), psnrs, color="teal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{noise_name} noise")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_metrics_bar.png"), dpi=150)
plt.close()

print("Done. Results written to", OUT)
for r in all_rows:
    print(r)
