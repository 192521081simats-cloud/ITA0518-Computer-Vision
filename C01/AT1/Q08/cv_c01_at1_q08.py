"""
Computer Vision - AT1 Short Answer Test
Question 8: Given a noisy image acquired from a sensor, explain how the
acquisition process contributes to noise and how it can be minimized.

Demo: add sensor-like noise (Gaussian thermal/shot noise + salt-and-pepper
hot-pixel noise) to a clean synthetic scene, then apply denoising filters
and compare PSNR before/after.

Run:  python3 q8.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q8_noise_and_denoising.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q8_noise_and_denoising():
    size = 256
    clean = np.zeros((size, size), dtype=np.float32)
    cv2.circle(clean, (size // 2, size // 2), 70, 1.0, -1)
    clean = cv2.GaussianBlur(clean, (0, 0), 3)  # soft object, like a real scene

    rng = np.random.default_rng(0)
    # Sensor noise sources: thermal/shot noise (Gaussian) + hot pixels (salt & pepper)
    gaussian_noise = rng.normal(0, 0.08, clean.shape)
    noisy = clean + gaussian_noise
    sp_mask = rng.random(clean.shape)
    noisy[sp_mask < 0.01] = 1.0
    noisy[sp_mask > 0.99] = 0.0
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    denoised_gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.2)
    denoised_median = cv2.medianBlur((noisy * 255).astype(np.uint8), 5).astype(np.float32) / 255

    def psnr(a, b):
        mse = np.mean((a - b) ** 2)
        return 999 if mse == 0 else 10 * np.log10(1.0 / mse)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    imgs = [clean, noisy, denoised_gaussian, denoised_median]
    titles = [
        "Clean scene",
        f"Acquired (noisy)\nPSNR={psnr(clean, noisy):.1f} dB",
        f"Gaussian filter\nPSNR={psnr(clean, denoised_gaussian):.1f} dB",
        f"Median filter\nPSNR={psnr(clean, denoised_median):.1f} dB",
    ]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q8_noise_and_denoising.png', dpi=160)
    plt.close()
    print("Q8: saved q8_noise_and_denoising.png "
          "(median filtering removes salt-and-pepper/hot-pixel noise "
          "better, Gaussian filtering smooths thermal/shot noise; PSNR "
          "quantifies the improvement)")


if __name__ == "__main__":
    q8_noise_and_denoising()
