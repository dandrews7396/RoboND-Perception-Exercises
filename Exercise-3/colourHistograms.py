#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
image = mpimg.imread('Udacican.jpeg')
# Your other options for input images are:
    # hammer.jpeg
    # beer.jpeg
    # bowl.jpeg
    # create.jpeg
    # disk_part.jpeg
    
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Convert from RGB to HSV
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ## Compute the histogram of the HSV channels
    h_hist = np.histogram(hsvImg[:,:,0], bins=nbins, range=bins_range)
    s_hist = np.histogram(hsvImg[:,:,1], bins=nbins, range=bins_range)
    v_hist = np.histogram(hsvImg[:,:,1], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    features_hist = np.concatenate((h_hist[0], s_hist[0], v_hist[0]))
    # Normalize the result
    norm_features = features_hist / np.sum(features_hist)
    # Return the feature vector
    return norm_features
    
feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
if feature_vec is not None:
    fig = plt.figure(figsize=(12,6))
    plt.plot(feature_vec)
    plt.title('HSV Feature Vector', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
else:
    print('Your function is returning None...')
