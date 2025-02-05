import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
image = cv2.imread('testimg.jpg')

# Memisahkan kanal warna
print(image)
channels = cv2.split(image)


colors = ('b', 'g', 'r')

for i, channel in enumerate(channels):
    print("channel", i)
    print(channel)


plt.figure(figsize=(10, 5))

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
# Menambahkan histogram dari ketiga warna
total_hist = np.zeros(32)
for channel in channels:
    total_hist += cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
total_hist /= 3
plt.plot(bin_edges, total_hist, color='k', linewidth=3, markersize=10, zorder=4, label='Total', 
            marker='o', markeredgecolor='black', alpha=0.5, drawstyle='default')

plt.title('Histogram Warna')
plt.xlabel('Intensitas')
plt.ylabel('Jumlah Piksel')
plt.legend()
plt.show()