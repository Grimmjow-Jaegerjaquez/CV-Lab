import cv2
import numpy as np

# Load the image
image = cv2.imread('resources\\image (19).jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(blurred_image, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for drawing contours
contour_image = image.copy()

# Define a list to store detected lesions
lesions = []

# Iterate through the detected contours
for contour in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(contour)

    # Set a minimum threshold for lesion size
    min_lesion_area = 5  # Adjust this threshold as needed
    max_lession_area = 50

    if area > min_lesion_area and area < max_lession_area:
        # Draw the contour on the original image
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

        # Get the coordinates and size of a bounding rectangle around the lesion
        x, y, w, h = cv2.boundingRect(contour)

        # Add the lesion coordinates and size to the list
        lesions.append((x, y, w, h))

# Display the image with detected lesions
cv2.imshow('Image with Lesions', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


