import cv2
import numpy as np
from skimage.exposure import cumulative_distribution
import os

def linear_relight(image):
  
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #make it grey

    cdfImageInput, binsImageInput = cumulative_distribution(grey_img)       #get cumdist
    cdfImageInput = np.insert(cdfImageInput, 0, [0]*binsImageInput[0])      #if first value not 0, fill it!
    cdfImageInput = np.append(cdfImageInput, [1]*(255-binsImageInput[-1]))  #if last value not 1, fill it!

    pixels = np.arange(256)                                         #make 256 long array 0-255 intensity
    linear_increase = np.linspace(0, 1, 256)                        #make 256 long array 0-1
    new_pixels = np.interp(cdfImageInput, linear_increase, pixels)  #interpolate intensity to fit linear
    
    imageOut = (np.reshape      #reshape back to 3D array
               (new_pixels[image.ravel()], image.shape)).astype(np.uint8)   
    #convert 1D > change pixel intensity to new_pixels > as bilangan bulat
    
    return imageOut



