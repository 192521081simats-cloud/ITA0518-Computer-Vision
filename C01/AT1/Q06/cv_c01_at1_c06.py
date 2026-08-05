"""
Computer Vision - AT1 Short Answer Test
Question 6: A camera captures... Explain how image formation models help
in improving image quality in applications such as autonomous driving.

Demo: pinhole-camera projection + lens-distortion correction (undistortion)
on a synthetic checkerboard scene (stand-in for a road/lane scene).

Run:  python3 q6.py
Requires: numpy, opencv-python (cv2), matplotlib
Output: q6_image_formation_model.png
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def q6_image_formation_model():
    # Build a synthetic checkerboard "scene" (stand-in for a road/lane scene)
    size = 400
    board = np.zeros((size, size), dtype=np.uint8)
    square = 40
    for r in range(0, size, square):
        for c in range(0, size, square):
            if ((r // square) + (c // square)) % 2 == 0:
                board[r:r + square, c:c + square] = 255
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

    # Camera intrinsic matrix (pinhole model): fx, fy, cx, cy
    fx = fy = 350
    cx, cy = size / 2, size / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # Real lens (radial) distortion coefficients that describe the camera
    dist_coeffs = np.array([-0.35, 0.15, 0, 0, 0], dtype=np.float64)

    # Build a synthetic "raw sensor capture" by warping the ideal checkerboard
    # with the (approximate) inverse of the distortion model, so that applying
    # cv2.undistort() with dist_coeffs afterwards genuinely straightens it back
    inv_coeffs = -dist_coeffs
    map_x, map_y = np.meshgrid(np.arange(size), np.arange(size))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    x_n = (map_x - cx) / fx
    y_n = (map_y - cy) / fy
    r2 = x_n**2 + y_n**2
    radial = 1 + inv_coeffs[0] * r2 + inv_coeffs[1] * r2**2
    map_x_d = (x_n * radial * fx + cx).astype(np.float32)
    map_y_d = (y_n * radial * fy + cy).astype(np.float32)
    distorted = cv2.remap(board_bgr, map_x_d, map_y_d, cv2.INTER_LINEAR)

    # Now correct it back using the camera's image formation model
    # (this is exactly what a self-driving car's vision pipeline must do
    # before lane/obstacle geometry can be trusted)
    undistorted = cv2.undistort(distorted, K, dist_coeffs)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, img, title in zip(
        axes,
        [board_bgr, distorted, undistorted],
        ["Ideal scene (pinhole model)",
         "Raw camera capture\n(lens distortion)",
         "Corrected using camera's\nimage formation model"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('q6_image_formation_model.png', dpi=160)
    plt.close()
    print("Q6: saved q6_image_formation_model.png "
          "(shows how modelling & correcting lens distortion restores "
          "straight lane/road geometry, critical for autonomous driving)")


if __name__ == "__main__":
    q6_image_formation_model()
