import cv2
from relight import linear_relight

image = cv2.imread("yourimage.jpg")
cv2.imshow('original', image)
imageOut = linear_relight(image)
cv2.imshow('linear', imageOut)

cv2.waitKey(0)
cv2.destroyAllWindows()