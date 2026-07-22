import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Generate a high-frequency grid pattern
img = np.zeros((200, 200), dtype=np.uint8)  # Create a blank black image array filled with zeros
img[::5, :] = 255
img[:, ::5] = 255

# Step 2: Apply a Gaussian Blur to act as an anti-aliasing (Low-Pass) filter
# The (5,5) kernel removes high frequencies before downsampling
blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian smoothing to reduce high-frequency noise

# Step 3: Now we can safely downsample the image
# cv2.INTER_AREA is mathematically designed to prevent aliasing
downsampled = cv2.resize(blurred, (40, 40), interpolation=cv2.INTER_AREA)  # Resize image to new dimensions using interpolation
bad_downsample = cv2.resize(img, (40, 40), interpolation=cv2.INTER_NEAREST)  # Resize image to new dimensions using interpolation

# Step 4: Show the difference
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(bad_downsample, cmap='gray'), plt.title('Aliased (Jagged)')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(downsampled, cmap='gray'), plt.title('Anti-Aliased (Smooth)')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
