import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an image
input_image = cv2.imread("resources\\cell.jpg", cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Apply edge detection (you can replace this with your preferred edge detection method)
edges = cv2.Canny(input_image, threshold1=50, threshold2=150)

# Apply Hough Line Transform
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=295)

# Plot the original image and detected lines
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edges")

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    plt.plot([x1, x2], [y1, y2], 'r')

plt.title("Detected Lines")
plt.show()