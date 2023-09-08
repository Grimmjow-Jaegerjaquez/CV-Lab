import cv2
import numpy as np

# read image
image = cv2.imread('resources\\download.png', cv2.IMREAD_UNCHANGED)
kernel1 = np.ones((5,5) , np.float32) / 30
img = cv2.filter2D(src = image, ddepth = -1, kernel = kernel1)

# apply guassian blur on src image
dst = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

# display input and output image
cv2.imshow("Original", image)
cv2.imshow("Gaussian Smoothing", dst)
cv2.imshow("Kernel Blur", img)
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image