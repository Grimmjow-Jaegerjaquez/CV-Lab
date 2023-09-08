import cv2
import numpy as np

# Load the image
image = cv2.imread('resources\\image (19).jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Flatten the grayscale image to a 1D array
pixels = gray_image.flatten().reshape((-1, 1))

# Define the number of clusters (K-means)
num_clusters = 10  # Adjust this based on your image and desired number of clusters

# Apply K-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reshape the labels to match the original image shape
segmented_image = labels.reshape(image.shape[:2])

# Define a threshold to identify potential lesion regions
lesion_label = 0  # You may need to adjust this label based on your K-means results

# Create a mask for potential lesion regions
lesion_mask = (segmented_image == lesion_label).astype(np.uint8)

# Apply post-processing to refine the lesion mask (e.g., morphological operations)

# Find contours in the lesion mask
contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for visualization
result_image = image.copy()

# Draw bounding boxes around detected lesions
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected lesions
cv2.imshow('Image with Lesions', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()