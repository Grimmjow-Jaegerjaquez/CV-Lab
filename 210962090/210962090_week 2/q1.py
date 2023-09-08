import cv2
import numpy as np

img = cv2.imread('resources\\image.jpg')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(grayimg)
res = np.hstack((grayimg, equ))
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()