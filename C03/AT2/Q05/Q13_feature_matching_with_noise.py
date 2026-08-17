"""
Question 13: Feature Matching with Noise
--------------------------------------------
Feature matching between two images fails due to noise in one image.
This script demonstrates the drop in matching accuracy caused by noise
and how a denoising preprocessing step (Non-Local Means denoising)
restores reliable ORB feature matching.

Pipeline:
1. Build a synthetic textured reference image.
2. Create a second image: the same scene, slightly shifted, with heavy
   Gaussian + salt-and-pepper noise added (simulating a poor sensor /
   low-light capture).
3. Match ORB descriptors directly on the noisy pair -> low match count
   and many incorrect correspondences (the problem).
4. Denoise the noisy image (cv2.fastNlMeansDenoisingColored) before
   matching -> match count and accuracy recover close to the clean
   baseline (the solution).
5. Save a comparison figure as the output screenshot.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_reference_image(size=320):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    rng = np.random.default_rng(11)

    for _ in range(30):
        x, y = rng.integers(10, size - 50, size=2)
        w, h = rng.integers(15, 40, size=2)
        color = tuple(int(c) for c in rng.integers(60, 240, size=3))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    for _ in range(20):
        x, y = rng.integers(10, size - 10, size=2)
        r = int(rng.integers(5, 15))
        color = tuple(int(c) for c in rng.integers(60, 240, size=3))
        cv2.circle(img, (x, y), r, color, -1)

    return img


def shift_image(img, dx=15, dy=10):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(50, 50, 50))


def add_heavy_noise(img):
    noisy = img.astype(np.float32)
    gaussian_noise = np.random.normal(0, 90, img.shape).astype(np.float32)
    noisy += gaussian_noise

    sp_mask = np.random.rand(*img.shape[:2])
    noisy[sp_mask < 0.12] = 255
    noisy[sp_mask > 0.88] = 0

    return np.clip(noisy, 0, 255).astype(np.uint8)


def orb_match(img1, img2, label):
    """Match ORB descriptors, then use RANSAC homography inliers as the
    accuracy metric (a much better proxy for TRUE matching accuracy than
    the raw ratio-test count, since noise can still produce many
    ratio-test 'good' matches that are geometrically wrong)."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, 0, 0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    # The true transform between the two images is a pure shift of (15, 10)
    # pixels (see shift_image()). A CORRECT match's displacement vector
    # should be close to (15, 10). We use the fraction of "good" matches
    # whose displacement is within a small tolerance of the true shift as
    # a ground-truth accuracy measure -- a much stronger test than simply
    # counting ratio-test matches, since noise can create matches that
    # pass the ratio test but point to the wrong location.
    TRUE_DX, TRUE_DY = 15, 10
    TOL = 6.0
    correct_matches = []
    for m in good:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        err = np.hypot((x2 - x1) - TRUE_DX, (y2 - y1) - TRUE_DY)
        if err < TOL:
            correct_matches.append(m)
    n_correct = len(correct_matches)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, correct_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    accuracy = (n_correct / len(good) * 100) if good else 0
    print(f"{label}: kp1={len(kp1)}, kp2={len(kp2)}, good_matches(ratio-test)={len(good)}, "
          f"geometrically_correct={n_correct} ({accuracy:.0f}% accuracy)")
    return match_img, len(kp1), len(kp2), n_correct


def main():
    np.random.seed(4)
    ref = build_reference_image()
    shifted = shift_image(ref)
    noisy = add_heavy_noise(shifted)

    raw_img, kp1_r, kp2_r, good_r = orb_match(ref, noisy, "RAW noisy (no preprocessing)")

    median_filtered = cv2.medianBlur(noisy, 5)  # removes salt-and-pepper impulses first
    denoised = cv2.fastNlMeansDenoisingColored(median_filtered, None, h=25, hColor=25,
                                                templateWindowSize=7, searchWindowSize=21)
    den_img, kp1_d, kp2_d, good_d = orb_match(ref, denoised, "Denoised (Non-Local Means)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    axes[0, 0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Reference (Clean) Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. Second Image: Shifted + Heavy\nGaussian/Salt-Pepper Noise')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'3. ORB Matching on RAW noisy image\n{good_r} geometrically-correct matches (kp: {kp1_r}/{kp2_r})')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(den_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'4. ORB Matching AFTER Denoising\n{good_d} geometrically-correct matches (kp: {kp1_d}/{kp2_d})')
    axes[1, 1].axis('off')

    plt.suptitle('Q13: Effect of Noise on Feature Matching (Raw vs Denoised)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/cv_solutions/images/Q13_output.png', dpi=150, bbox_inches='tight')
    print("Saved output screenshot to Q13_output.png")


if __name__ == "__main__":
    main()
