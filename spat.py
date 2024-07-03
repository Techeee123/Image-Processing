import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('wolf.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Downsample the image by a factor of 4
downsample_factor = 4
low_res_image = image[::downsample_factor, ::downsample_factor]

# Display the low-resolution image
plt.subplot(1, 3, 2)
plt.title('Low-Resolution Image')
plt.imshow(low_res_image, cmap='gray')
plt.axis('off')

# Upsample the image back to original size using nearest-neighbor interpolation
upsampled_image = cv2.resize(low_res_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Display the upsampled image
plt.subplot(1, 3, 3)
plt.title('Upsampled Image')
plt.imshow(upsampled_image, cmap='gray')
plt.axis('off')

# Show the plots
plt.show()
