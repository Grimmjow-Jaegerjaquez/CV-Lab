import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class KMeans:
    def __init__(self, num_clusters, max_iters=100):
        self.num_clusters = num_clusters
        self.max_iters = max_iters

    def fit(self, data):
        self.centroids = data[np.random.choice(len(data), self.num_clusters, replace=False)]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.num_clusters)])
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels


def color_segmentation(image, num_segments=3):
    # Reshape the image to a 2D array of RGB values
    height, width, channels = image.shape
    pixels = image.reshape((height * width, channels))

    # Perform K-Means clustering
    kmeans = KMeans(num_clusters=num_segments)
    labels = kmeans.fit(pixels)

    # Create a segmented image based on cluster labels
    segmented_image = np.zeros_like(pixels)
    for segment_id in range(num_segments):
        segment_mask = labels == segment_id
        segmented_image[segment_mask] = kmeans.centroids[segment_id]

    segmented_image = segmented_image.reshape((height, width, channels))
    return segmented_image


# Load an image
input_image = Image.open("resources\\srk.jfif")

# Convert the image to a numpy array
image_array = np.array(input_image)

# Perform color segmentation
segmented_image = color_segmentation(image_array, num_segments=4)

# Display the original and segmented images
plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("Segmented Image")

plt.show()