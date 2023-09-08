import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
input_image = cv2.imread("resources\\srk.jfif")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Define color ranges for segmentation (in RGB format)
lower_bound = np.array([100, 100, 100])
upper_bound = np.array([200, 150, 150])

# Create a mask based on color thresholding
mask = cv2.inRange(input_image, lower_bound, upper_bound)

# Apply the mask to the original image
segmented_image = cv2.bitwise_and(input_image, input_image, mask=mask)

# Display the original and segmented images
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("Segmented Image")

plt.show()