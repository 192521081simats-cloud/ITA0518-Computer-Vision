"""
Computer Vision - AT2 Concept Mapping
Question 9: Design a concept map that identifies the role of image
sensing mechanisms, clearly establishing relationships between sensor
characteristics, noise generation, and acquisition quality, integrating
literature perspectives, and ensuring logical flow.

Demo: apply three noise sources associated with real image sensors
(Gaussian read noise, Poisson/shot noise, salt-and-pepper defects) to a
clean image and quantify the resulting drop in acquisition quality using
PSNR (peak signal-to-noise ratio).

Run:  python3 cv_co1_at2_q9.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q9_sensing_mechanisms_noise_and_quality.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def psnr(clean, noisy):
    mse = np.mean((clean.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_poisson_noise(img, scale=0.05):
    vals = img.astype(np.float64) * scale
    noisy = np.random.poisson(vals) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.03):
    out = img.copy()
    mask = np.random.rand(*img.shape)
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def q9_sensing_mechanisms_noise_and_quality():
    size = 256
    clean = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(clean, (50, 50), (200, 200), 200, -1)
    cv2.circle(clean, (128, 128), 50, 90, -1)

    variants = [
        ("Clean (ideal sensor)", clean),
        ("Read noise\n(Gaussian, sensor electronics)", add_gaussian_noise(clean)),
        ("Shot noise\n(Poisson, photon statistics)", add_poisson_noise(clean)),
        ("Defective pixels\n(salt-and-pepper)", add_salt_pepper_noise(clean)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, img) in zip(axes, variants):
        quality = psnr(clean, img)
        quality_str = "reference" if quality == float('inf') else f"PSNR: {quality:.1f} dB"
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{title}\n{quality_str}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('q9_sensing_mechanisms_noise_and_quality.png', dpi=160)
    plt.close()
    print("Q9: saved q9_sensing_mechanisms_noise_and_quality.png "
          "(different sensing mechanisms generate different characteristic "
          "noise - electronic read noise, photon shot noise, or defective "
          "pixels - and each measurably lowers acquisition quality, here "
          "quantified with PSNR)")


if __name__ == "__main__":
    q9_sensing_mechanisms_noise_and_quality()
