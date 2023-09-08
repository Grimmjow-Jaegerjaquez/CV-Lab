import numpy as np
import matplotlib.pyplot as plt
import cv2


def hough_transform(image):
    height, width = image.shape
    max_rho = int(np.sqrt(height ** 2 + width ** 2))  # Maximum possible distance from the origin
    thetas = np.deg2rad(np.arange(-90, 90))  # Range of theta values in radians
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((max_rho * 2, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(image)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int(x * cos_thetas[t_idx] + y * sin_thetas[t_idx])
            accumulator[rho + max_rho, t_idx] += 1

    return accumulator, thetas


def hough_lines(image, accumulator, thetas, threshold):
    lines = []
    rhos, theta_idxs = np.where(accumulator > threshold)

    for i in range(len(rhos)):
        rho = rhos[i] - len(accumulator) // 2
        theta_idx = theta_idxs[i]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append(((x1, y1), (x2, y2)))

    return lines


# Read an image and convert it to grayscale
image = cv2.imread('resources\\images.jfif', cv2.IMREAD_GRAYSCALE)

# Apply edge detection (Canny) to the image
edges = cv2.Canny(image, threshold1=50, threshold2=150)

# Perform the Hough Transform
accumulator, thetas = hough_transform(edges)

# Set a threshold and get the detected lines
threshold = 100
detected_lines = hough_lines(edges, accumulator, thetas, threshold)

# Plot the original image and detected lines
plt.imshow(image, cmap='gray')
for line in detected_lines:
    plt.plot(*zip(*line), color='red')
plt.show()