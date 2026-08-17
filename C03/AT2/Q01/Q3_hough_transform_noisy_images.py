"""
Question 3: Hough Transform in Noisy Images
--------------------------------------------
A system uses Hough Transform to detect lines in noisy images but produces
inaccurate results. This script demonstrates the problem and the fix.

Pipeline:
1. Create a clean synthetic image with straight lines.
2. Add Gaussian + salt-and-pepper noise to simulate a noisy sensor image.
3. Run Canny + Hough Transform directly on the noisy image  -> many
   spurious / broken line detections (the problem).
4. Apply preprocessing (median blur to kill salt-and-pepper noise,
   Gaussian blur to smooth remaining noise) and re-tune the Canny /
   Hough parameters -> clean, accurate line detection (the solution).
5. Save a 4-panel comparison figure as the output screenshot.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_clean_line_image(size=500):
    """Create a synthetic image containing a few straight lines."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.line(img, (50, 50), (450, 50), 255, 3)
    cv2.line(img, (50, 50), (50, 450), 255, 3)
    cv2.line(img, (60, 400), (440, 100), 255, 3)
    cv2.line(img, (100, 450), (450, 450), 255, 3)
    return img


def add_noise(img):
    """Add Gaussian noise + salt-and-pepper noise to simulate a noisy capture."""
    noisy = img.astype(np.float32)

    # Gaussian noise
    gaussian_noise = np.random.normal(0, 35, img.shape).astype(np.float32)
    noisy = noisy + gaussian_noise

    # Salt-and-pepper noise
    salt_pepper_mask = np.random.rand(*img.shape)
    noisy[salt_pepper_mask < 0.02] = 255
    noisy[salt_pepper_mask > 0.98] = 0

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def detect_lines_naive(noisy_img):
    """Run Hough Transform directly on the noisy image (no preprocessing)."""
    edges = cv2.Canny(noisy_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=30, maxLineGap=5)
    result = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
    n_lines = 0 if lines is None else len(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return edges, result, n_lines


def detect_lines_improved(noisy_img):
    """Preprocess the noisy image, then run Hough Transform with tuned params."""
    # Median blur removes salt-and-pepper noise very effectively
    denoised = cv2.medianBlur(noisy_img, 5)
    # Gaussian blur smooths remaining Gaussian noise before edge detection
    denoised = cv2.GaussianBlur(denoised, (5, 5), 1.2)

    edges = cv2.Canny(denoised, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=80, maxLineGap=10)
    result = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
    n_lines = 0 if lines is None else len(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return denoised, edges, result, n_lines


def main():
    np.random.seed(7)
    clean = build_clean_line_image()
    noisy = add_noise(clean)

    naive_edges, naive_result, n_naive = detect_lines_naive(noisy)
    denoised, imp_edges, imp_result, n_improved = detect_lines_improved(noisy)

    print(f"Lines detected WITHOUT preprocessing: {n_naive}")
    print(f"Lines detected WITH preprocessing:    {n_improved}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('1. Clean Synthetic Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('2. Noisy Image\n(Gaussian + Salt-Pepper Noise)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(naive_edges, cmap='gray')
    axes[0, 2].set_title('3. Canny Edges (No Preprocessing)\nNoisy, fragmented edges')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(naive_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'4. Hough Result - NO Preprocessing\n{n_naive} spurious lines detected')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(imp_edges, cmap='gray')
    axes[1, 1].set_title('5. Canny Edges (After Median +\nGaussian Blur Preprocessing)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(imp_result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'6. Hough Result - WITH Preprocessing\n{n_improved} accurate lines detected')
    axes[1, 2].axis('off')

    plt.suptitle('Q3: Effect of Noise on Hough Transform Line Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/cv_solutions/images/Q3_output.png', dpi=150, bbox_inches='tight')
    print("Saved output screenshot to Q3_output.png")


if __name__ == "__main__":
    main()
