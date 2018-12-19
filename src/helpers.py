#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:12:58 2018

@author: roshanprakash
"""

import os
import sys
import pickle
import klepto
import numpy as np
import gc

def write_to_file(obj, filepath, var_name = None, key = None):
    """
    Writes action sequence object to file.
    """
    d = klepto.archives.dir_archive(filepath, cached=True, serialized=True)
    d.load(var_name)
    if var_name in d:
        if key in d[var_name]:
            d[var_name][key] = np.concatenate((d[var_name][key], obj))
        else:
            d[var_name][key] = obj
    else:
        d[var_name] = {}
        d[var_name][key] = obj
    d.sync()
    d.clear()
    gc.collect()


def load_from_file(filepath, var_name = None):
    """
    Loads action sequence object from file.
    """
    d = klepto.archives.dir_archive(filepath, cached=True, serialized=True)
    d.load(var_name)
    obj = d[var_name]
    d.clear()
    return obj


def get_batches(X_train, y_train, optical_flows_train, motion_representations_train):
    
    """ Returns a batch of training data 
        param X_train : the training inputs of shape (N, 120, 120, 1)
        param y_train : the training targets of shape (N, 120, 120, 1)
        param optical_flows_train : the optical_flows data of shape (N, 120, 120, 1)
        param motion_representations_train : the motion_representations data of shape (N, 120, 120, 1)
        returns 16-sized batches of shuffled training images(X), target_images(y), corresponding optical_flow_inputs(OPF) and corresponding future motion representations(MM).
    """

    N, H, W, C = X_train.shape
    
    num_batches = int(N / 16.0) # will be perfectly divisible . check dataset loader for details!
    
    idxs = np.random.shuffle(np.arange(N))
    
    X = X_train[idxs]#[0]
    X = np.reshape(X, (num_batches, 16, H, W, C))
    
    y = y_train[idxs]
    y = np.reshape(y, (num_batches, 16, H, W, C))
    
    OPF = optical_flows_train[idxs] 
    OPF = np.reshape(OPF, (num_batches, 16, H, W, C)) 
    
    MM = motion_representations_train[idxs] 
    MM = np.reshape(MM, (num_batches, 16, H, W, C)) 
       
    return X, y, OPF, MM
