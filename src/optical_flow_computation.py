#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:45:01 2018

@author: roshanprakash
"""

import cv2 as cv
import numpy as np

def compute_optical_flow(x):
    
    """ Computes a multi-layered dense optical flow output from a sequence of images first. \
            Next, the average optical flow from this dense flow output will be computed. \
            Finally, transforms the average flow output to a dimension identical to that of each input image.
        param x : a numpy array of a sequence of images, shape --> (N, H, W, 3)
        returns : a numpy array of shape (1, H, W) corresponding to the average optical flow
    """
    
    try:
        N, H, W, C = x.shape
    
    except:
        raise ValueError('Check the inputs! Something went wrong there.')
       
    # initializations
    hsv = np.zeros_like(x[0])
    dense_optical_flow = []
    
    # first compute a dense multi-layered optical flow from the sequence of image frames
    for idx in range(1, x.shape[0]): # start from the second image to compute pair-wise optical flow between successive images
        
        # note here that we are already using grayscale images of shape (N, H, 3) \
        # so we will only need pixels from one channel as all the channels have the same corresponding pixel values ! 
        prev_img = x[idx - 1, :][:, :, 0]
        next_img = x[idx, :][:, :, 0]
        flow = cv.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0) # has shape (H, W, 2)
        dense_optical_flow.append(flow)
     
    # next, compute the average flow 
    average_flow = np.mean(np.array(dense_optical_flow), axis = 0, keepdims = False)
    
    # next, map from cartesian coordinates to polar coordinates
    magnitude, angle = cv.cartToPolar(average_flow[..., 0], average_flow[..., 1])
    
    # next, map from polar coordinates to HSV coordinates
    hsv[..., 0] = angle * 180 / (np.pi / 2) # angle ---> hue component
    hsv[..., 1] = 255 # all pixels are given full saturation
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) # magnitude ----> value component
    
    # next, map from HSV to RGB space
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    
    # finally, mapping from RGB to grayscale 
    optical_flow = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY).reshape((1, H, W))
    
    return optical_flow
