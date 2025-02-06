import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution


image = cv2.imread('testimg.jpg')
channels = cv2.split(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cdfImageInput, binsImageInput = cumulative_distribution(gray_image)
plt.figure(figsize=(10, 5))
plt.plot(binsImageInput, cdfImageInput)
plt.title('cumulative gray')



plt.figure(figsize=(10, 5))
avg_hist = np.zeros(256)
colors = ('b', 'g', 'r')
for (channel, color) in zip(channels,colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 255])
    avg_hist+=hist.flatten()
    print("hist {color}", hist)
    plt.plot(hist, color=color)
avg_hist/=3

plt.plot(avg_hist, color='y')
plt.title("histogram RGB")

plt.figure(figsize=(10, 5))
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 255])
plt.plot(hist, color='k')
plt.title("histogram Grey")



print(f"First bins: {binsImageInput[0]}, Cumulative Probability: {cdfImageInput[0]:.5f}")
print(f"Last bins: {binsImageInput[-1]}, CUmulative Probability: {cdfImageInput[-1]:.5f}")
cdfImageInput = np.insert(cdfImageInput, 0, [0]*binsImageInput[0])
cdfImageInput = np.append(cdfImageInput, [1]*(255-binsImageInput[-1]))

histconvert = []
for i, value in enumerate(cdfImageInput):
    if i==0:
        histconvert.append(value)
    else:
        histconvert.append(value-cdfImageInput[i-1])
plt.figure(figsize=(10, 5))
plt.plot(histconvert, color='k')
plt.title("histogram Covert")

total_avg_hist = np.sum(avg_hist)
print(f"Total value of avg_hist: {total_avg_hist}")
for i, value in enumerate(avg_hist):
    value /= total_avg_hist
    avg_hist[i]=value
plt.figure(figsize=(10, 5))
plt.plot(avg_hist, color='k')
plt.title("histogram avg rgb normalized")

for i, value in enumerate(avg_hist):
    if i>0:
        avg_hist[i]=value+avg_hist[i-1] 

plt.figure(figsize=(10, 5))
plt.plot(avg_hist, color='k')
plt.title("cumulative avg color")

height, width, _ = image.shape
image_size = height * width
print(f"Image size = {image_size}")

pixels = np.arange(256)




plt.show()