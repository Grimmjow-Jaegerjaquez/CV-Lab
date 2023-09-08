import numpy as np
import cv2


def compute_lbp_pixel(center, pixels):
    lbp_code = 0
    for i, pixel in enumerate(pixels):
        if pixel >= center:
            lbp_code |= (1 << i)
    return lbp_code


def lbp_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_result = np.zeros_like(gray_image)

    rows, cols = gray_image.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = gray_image[r, c]
            pixels = [
                gray_image[r - 1, c - 1], gray_image[r - 1, c], gray_image[r - 1, c + 1],
                gray_image[r, c - 1], gray_image[r, c + 1],
                gray_image[r + 1, c - 1], gray_image[r + 1, c], gray_image[r + 1, c + 1]
            ]

            lbp_code = compute_lbp_pixel(center, pixels)
            lbp_result[r, c] = lbp_code

    return lbp_result


# Load an image
image = cv2.imread('resources\\images.jfif')

# Compute LBP image
lbp_result = lbp_image(image)

# Display the original and LBP images
cv2.imshow('Original Image', image)
cv2.imshow('LBP Image', lbp_result)
cv2.waitKey(0)
cv2.destroyAllWindows()