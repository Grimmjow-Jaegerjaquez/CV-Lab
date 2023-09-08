import cv2
import matplotlib.pyplot as plt

source = cv2.imread("resources\\lean.jpg")
source = cv2.resize(source, (0, 0), fx = 0.4, fy = 0.4)

img = cv2.Laplacian(source, cv2.CV_16S, ksize=3)
abs_img = cv2.convertScaleAbs(img)

cv2.imshow("Edge", abs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()