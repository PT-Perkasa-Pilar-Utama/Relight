import cv2
from skimage import exposure


dst = cv2.imread("terang.jpg")
src = cv2.imread("gelap.jpg")
matched_src = exposure.match_histograms(src,dst, channel_axis=2)
cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('result', matched_src)

cv2.waitKey(0)
cv2.destroyAllWindows()