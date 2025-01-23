import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import *
from skimage import exposure
import os
import glob

def load_image(filename):
  img = cv2.imread(filename)
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return rgb

def show_image(image):
  plt.figure(figsize=(12,5))
  plt.imshow(image)
  plt.axis('off')
  plt.show()

def calculate_hist(image):
  fig,ax = plt.subplots(1,2,figsize=(15,5))
  ax[0].imshow(image)
  ax[0].axis('off')
  ax[0].set_title("Image")

  ax[1].hist(image.ravel(), bins=32, range=(0.0, 256.0), ec='k') #calculating histogram
  ax[1].set_title("Histogram")
  ax[1].set_xlabel("range")
  plt.show()

def compare_matched_hist(src,dst,matched_src):
  images = [src,dst,matched_src]
  headings = ["Source","Destination","Matched Source"]
  n,m = len(images),2
  fig,ax = plt.subplots(n,m,figsize=(15,10))

  for i, (heading,img) in enumerate(zip(headings,images)):
    ax[i,0].imshow(img)
    ax[i,0].axis('off')
    ax[i,0].set_title(heading)

    ax[i,1].hist(img.ravel(), bins=32, range=(0.0, 256.0), ec='k') #calculating histogram
  plt.show()


def histogram_matching(folder_path, output_folder, dst_path):
    for f in glob.glob(folder_path + "/*.jpg"):
        print(f)
        dst = load_image(dst_path)
        src = load_image(f)
        matched_src = exposure.match_histograms(src,dst, channel_axis=2)

        file_name = f.split('/')[-1].split('.')[0]
        output_path = os.path.join(output_folder, '{}.jpg'.format(file_name))
        plt.imsave(output_path, matched_src)    
