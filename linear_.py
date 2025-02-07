import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution
import os

def linear_relight(image_path):
    print(image_path)
    linear_increase = np.linspace(0, 1, 256)
    plt.figure(figsize=(10, 5))
    plt.plot(linear_increase, color='k')
    plt.title("Linear vs cumdist")

    image = cv2.imread(image_path)
    channels = cv2.split(image)
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cdfImageInput, binsImageInput = cumulative_distribution(grey_img)
    print(len(cdfImageInput))
    cdfImageInput = np.insert(cdfImageInput, 0, [0]*binsImageInput[0])
    cdfImageInput = np.append(cdfImageInput, [1]*(255-binsImageInput[-1]))
    print(len(cdfImageInput))
    plt.plot(cdfImageInput, color='r')

    pixels = np.arange(256)
    new_pixels = np.interp(cdfImageInput, linear_increase, pixels)
    
    

    imageOut = (np.reshape(new_pixels[image.ravel()], image.shape)).astype(np.uint8)
    grey_img = cv2.cvtColor(imageOut, cv2.COLOR_BGR2GRAY)
    cdfImageInput, binsImageInput = cumulative_distribution(grey_img)
    plt.plot(cdfImageInput, color='b')  


    cv2.imshow('imageOut.jpg', imageOut)
    cv2.imshow('image.jpg', image)
    
    cv2.waitKey(0)


image_folder = "train/images"
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        linear_relight(image_path)
