import cv2
import numpy as np

img = cv2.imread("resources\\lean.jpg")
img = cv2.resize(img, (0, 0), fx = 0.4, fy = 0.4)
can = cv2.Canny(img, 90, 110, L2gradient=True)
cv2.imshow("Original", img)
cv2.imshow("lena", can)
cv2.waitKey(0)
cv2.destroyAllWindows()