import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution


# Membaca gambar
image = cv2.imread('testimg.jpg')
channels = cv2.split(image)
colors = ('b', 'g', 'r')

for i, channel in enumerate(channels):
    print("channel", i)
    print(channel)


plt.figure(figsize=(10, 5))

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate cumulative distribution for grayscale image
cdfImageInput, binsImageInput = cumulative_distribution(image)
histotest = []
for i, value in enumerate(cdfImageInput):
    if i>0:
        histotest.append(cdfImageInput[i]-cdfImageInput[i-1])
    else:
        histotest.append(cdfImageInput[i])
print("histotest", histotest)  
# Multiply each value in histotest by 255
histotest = [value * 100 for value in histotest]
# Calculate histogram for grayscale image
hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

pixels = np.arange(256)
new_pixels = np.interp(cdfImageInput, cdfImageInput, pixels)
plt.plot(binsImageInput, new_pixels,linewidth=5)
plt.plot(binsImageInput, histotest, linewidth=5)
plt.plot(binsImageInput, cdfImageInput, linewidth=5)
plt.xlim(0, 255)
plt.ylim(0, 1)
plt.xlabel('Pixel Values')
plt.ylabel('Cumulative Probability')





# Plot histogram for grayscale image
plt.figure(figsize=(10, 5))
plt.plot(hist_gray, color='k', linewidth=2)
plt.xlim([0, 256])
plt.title('Histogram Grayscale Image')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.show()


print(f"First bins: {binsImageInput[0]}, Cumulative Probability: {cdfImageInput[0]:.5f}")
print(f"Last bins: {binsImageInput[-1]}, CUmulative Probability: {cdfImageInput[-1]:.5f}")
cdfImageInput = np.insert(cdfImageInput, 0, [0]*binsImageInput[0]) # fill 0 in index 0 - 17
cdfImageInput = np.append(cdfImageInput, [1]*(255-binsImageInput[-1])) # fill 1 in index 247 - 255

# Create a second figure window
plt.figure(figsize=(10, 5))


# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# plt.show()



















# Membuat histogram untuk setiap kanal warna
total_hist = np.zeros(32)
for i, (channel, color) in enumerate(zip(channels, colors)):
    hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
    
    total_hist += hist.flatten()
    bin_edges = np.arange(32) * 8
    plt.plot(bin_edges, hist, color=color, linewidth = 3, markersize =10, zorder=(3-i), label = color,
             marker = 'o', markeredgecolor='black', alpha=1, drawstyle = 'default')
    plt.xlim([0, 256])
plt.plot(bin_edges, total_hist, color='y', linewidth=3, markersize=10, zorder=4, label='Total 0',)

total_hist /= 3
plt.plot(bin_edges, total_hist, color='k', linewidth=3, markersize=10, zorder=4, label='Total', 
            marker='o', markeredgecolor='black', alpha=0.5, drawstyle='default')


height, width = image.shape[:2]
new_width = 400
new_height = int((new_width / width) * height)
resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow('Original Image', resized_image)

plt.title('Histogram Warna')
plt.xlabel('Intensitas')
plt.ylabel('Jumlah Piksel')
plt.legend()
plt.show()