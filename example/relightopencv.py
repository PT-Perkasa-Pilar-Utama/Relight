import cv2
from linear import linear_relight
from skimage import exposure


image = cv2.imread("development/gelap.jpg")
cv2.imshow('original', image)
imageOut = linear_relight(image)
cv2.imshow('linear', imageOut)

cv2.waitKey(0)
cv2.destroyAllWindows()