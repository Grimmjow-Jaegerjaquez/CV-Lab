import numpy as np
import cv2


def harris_corner_detection(image, k=0.04, window_size=3, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Calculate gradients using Sobel operators
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate products of gradients for the structure tensor
    dx2 = dx ** 2
    dy2 = dy ** 2
    dxy = dx * dy

    # Apply Gaussian blur to the products of gradients
    dx2 = cv2.GaussianBlur(dx2, (window_size, window_size), 0)
    dy2 = cv2.GaussianBlur(dy2, (window_size, window_size), 0)
    dxy = cv2.GaussianBlur(dxy, (window_size, window_size), 0)

    # Calculate Harris response
    det = dx2 * dy2 - dxy ** 2
    trace = dx2 + dy2
    harris_response = det - k * (trace ** 2)

    # Normalize and threshold the Harris response
    harris_response[harris_response < threshold * harris_response.max()] = 0

    # Find corner points
    corner_points = np.argwhere(harris_response > 0)

    return corner_points


# Load an image
image = cv2.imread('resources\\images.jfif')

# Detect corners using Harris Corner Detection
corner_points = harris_corner_detection(image)

# Draw circles at detected corner points
for point in corner_points:
    cv2.circle(image, tuple(point[::-1]), 3, (0, 255, 0), -1)

# Display the image with detected corners
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()