#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:37:56 2018

@author: roshanprakash
"""

import os
import cv2 as cv
import math
import numpy as np
import pickle 
import logging
from optical_flow import *

import logging
logging.basicConfig(level = logging.INFO)
data_logger = logging.getLogger(__name__ + '.data')


class dataset_loader():
    
    def __init__(self, delta_t, k, offset):

        """ *arg data_dir : the source directory for the data, as a pathname ; type --> str
            
            *arg delta_t  : the difference in timesteps(interpreted as milliseconds) between 
                            any input image and the desired target image; type --> float
            
            *arg k : the number of previous(sequential) images to be used while computing the optical flow for any input image.
                        (should be fixed as a constant with this initialization) ; type --> int
            
            *arg offset : a value indicating the number of images to skip from every image sequence directory.
                        i.e, for every 'n' images in any sequence directory, images(1,..k) and images(n-k,...n) 
                        will be skipped. (to ignore images which do not contain potential motion information)   
        """
        
        self.path =  os.path.dirname(os.getcwd()) + '/datasets' # --> root directory + '/datasets'
        #self.path = '../datasets' while running from a script
        self.delta_t = delta_t
        self.k = k
        self.offset = offset
        self.X = {}
        self.y = {}
        self.optical_flows = {}
        self.motion_representations = {}
        self.means = {}
        self.stdevs = {}
        self.load_data()
        
    
    def load_data(self):
        
        """ Loads the images from the data source and generates training and testing datasets."""
        
        load_path = os.path.dirname(os.getcwd()) + '/pickles'
        #load_path = '../pickles' while running from a script
        X_train_path = load_path + '/X_train_dt_{}.pkl'.format(int(self.delta_t))
        y_train_path = load_path + '/y_train_dt_{}.pkl'.format(int(self.delta_t))
        X_test_path = load_path + '/X_test_dt_{}.pkl'.format(int(self.delta_t))
        y_test_path = load_path + '/y_test_dt_{}.pkl'.format(int(self.delta_t))
        optical_flows_train_path = load_path + '/optical_flows_train_dt_{}.pkl'.format(int(self.delta_t))
        optical_flows_test_path = load_path + '/optical_flows_test_dt_{}.pkl'.format(int(self.delta_t))
        motion_representations_train_path = load_path + '/motion_representations_train_dt_{}.pkl'.format(int(self.delta_t))
        
        # First, lookup if pickled data is available for loading
        if os.path.exists(X_train_path) and os.path.exists(y_train_path) and os.path.exists(X_test_path) and  os.path.exists(y_test_path) \
                and os.path.exists(optical_flows_train_path) and  os.path.exists(optical_flows_test_path) and  os.path.exists(motion_representations_train_path):
            
            data_logger.info('Pickled data available. Loading now!')
            
            # all required data were already pickled. Just load them here!      
            with open(X_train_path, 'rb') as xtr:
                self.X_train = pickle.load(xtr)
                
            with open(y_train_path, 'rb') as ytr:
                self.y_train = pickle.load(ytr)
                
            with open(X_test_path, 'rb') as xte:
                self.X_test = pickle.load(xte)
            
            with open(y_test_path, 'rb') as yte:
                self.y_test = pickle.load(yte)
            
            with open(optical_flows_train_path, 'rb') as optr:
                self.optical_flows_train = pickle.load(optr)
                
            with open(optical_flows_test_path, 'rb') as opte:
                self.optical_flows_test = pickle.load(opte)
                
            with open(motion_representations_train_path, 'rb') as mtr:
                self.motion_representations_train = pickle.load(mtr)
                
                
        else:# no pickled data found. Need to gather data from scratch!
            
            data_logger.info('No pickled data found. Starting to load new data! This may take a while.')
            
            # initialization
            self.X_train = {}
            self.y_train = {}
            self.X_test = {}
            self.y_test = {}    
            self.optical_flows_train = {}
            self.optical_flows_test = {}
            self.motion_representations_train = {}
            
            # structure of main images dataset : datasets --> action_type_directory ---> multiple sequence sub-directories ---> multiple images 
            
            # for every action-type directory in the main dataset
            for action_dir_name in os.listdir(self.path):
        
                if action_dir_name == '.DS_Store':  
                    
                    pass
                
                else: 
                    
                    if not action_dir_name in self.X.keys():
                        self.X[action_dir_name] = []
                    if not action_dir_name in self.y.keys():
                        self.y[action_dir_name] = []
                    if not action_dir_name in self.optical_flows.keys():
                        self.optical_flows[action_dir_name] = []
                    if not action_dir_name in self.motion_representations.keys():
                        self.motion_representations[action_dir_name] = []
                    
                    action_path = os.path.join(self.path, action_dir_name)
                    
                    # for every sequence directory within the action directory
                    for seq_name in os.listdir(action_path):
                        
                        images = [] # reset for every sequence of images since we will be gathering all inputs in a per-sequence directory manner
                        
                        if seq_name == '.DS_Store':
                            pass
                        
                        else:
                            seq_path = os.path.join(action_path, seq_name)
                            for img_name in sorted(os.listdir(seq_path), key=lambda filename: int(filename[:-4])):
                                images.append(cv.imread(os.path.join(seq_path, img_name)))
        
                        seq_size = len(images)
                        
                        if not seq_size > self.k + self.delta_t + (2 * self.offset):
                            data_logger.info('Not enough images in this sequence directory to build training and test images. So, ignoring this sequence directory for now!')
                        
                        else: # we have enough images to build model training and test data
                            
                            for img_idx in range(self.k + self.offset, seq_size - self.offset):
                                # note here that we will need images from kth frame onward for optical flow inputs to be available as well !
                                # also note that we will only require images for which targets(img[current_time_step + delta_t]) exist !
                                if img_idx + self.delta_t < seq_size - self.offset:
                                    self.X[action_dir_name].append(images[img_idx])
                                    self.y[action_dir_name].append(images[img_idx + int(self.delta_t)])
                                    # (below) calculates and gathers the optical flow input for this image frame using 'k' previous image frames
                                    self.optical_flows[action_dir_name].append(compute_optical_flow(np.array(images[img_idx - self.k : img_idx + 1]))) 
                                    # (below) calculates and gathers the 'motion representation' for this image frame using all sequential future image frames until target image
                                    self.motion_representations[action_dir_name].append(compute_optical_flow(np.array(images[img_idx: img_idx + int(self.delta_t)+ 1])))
                
                            data_logger.info('Sequence loading complete!')
                        
                    # Now, we have X, y and other motion information from this action directory 
                    # Note that action directories have unique pathnames. So, no overwriting or any loss of information will happen in the future, after the below steps
                    self.X[action_dir_name] = np.asarray(self.X[action_dir_name])
                    self.y[action_dir_name] = np.asarray(self.y[action_dir_name])
                    self.optical_flows[action_dir_name] = np.asarray(self.optical_flows[action_dir_name])
                    self.motion_representations[action_dir_name] = np.asarray(self.motion_representations[action_dir_name])
                
            
                    ########################### Split into training and testing datasets ################################
            
                    # Shuffle the inputs and normalize
                    idxs = np.random.shuffle(np.arange(self.X[action_dir_name].shape[0])) 
                    self.X[action_dir_name] = self.X[action_dir_name][idxs][0] / 255.0
                    self.y[action_dir_name] = self.y[action_dir_name][idxs][0] / 255.0
                    self.optical_flows[action_dir_name] = self.optical_flows[action_dir_name][idxs][0] # already normalized
                    self.motion_representations[action_dir_name] = self.motion_representations[action_dir_name][idxs][0] # already normalized
                    
                    N = self.X[action_dir_name].shape[0]
                    
                    # Notice here that we will need images of shape --> (N, H, W, 1)
                    self.X_train[action_dir_name] = self.X[action_dir_name][:math.floor(0.8 * N)][: , : , :, [0]]
                    self.y_train[action_dir_name] = self.y[action_dir_name][:math.floor(0.8 * N)][: , : , :, [0]]
                    self.X_test[action_dir_name] = self.X[action_dir_name][math.floor(0.8 * N):][: , : , :, [0]]
                    self.y_test[action_dir_name] = self.y[action_dir_name][math.floor(0.8 * N):][: , : , :, [0]]
                    self.optical_flows_train[action_dir_name] = self.optical_flows[action_dir_name][:math.floor(0.8 * N)]
                    self.optical_flows_test[action_dir_name] = self.optical_flows[action_dir_name][math.floor(0.8 * N):]
                    # we will not require motion representation inputs during test time !
                    self.motion_representations_train[action_dir_name] = self.motion_representations[action_dir_name][:math.floor(0.8 * N)] 
                    
            # Lastly, pickle the data for later reuse
            
            save_path = load_path 
            
            with open(save_path + '/X_train_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as xtr:
                pickle.dump(self.X_train, xtr, protocol = pickle.HIGHEST_PROTOCOL)
    
            with open(save_path + '/y_train_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as ytr:
                pickle.dump(self.y_train, ytr, protocol = pickle.HIGHEST_PROTOCOL)
                
            with open(save_path + '/X_test_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as xte:
                pickle.dump(self.X_test, xte, protocol = pickle.HIGHEST_PROTOCOL)
                    
            with open(save_path + '/y_test_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as yte:
                pickle.dump(self.y_test, yte, protocol = pickle.HIGHEST_PROTOCOL)
                    
            with open(save_path + '/optical_flows_train_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as oftr:
                pickle.dump(self.optical_flows_train, oftr, protocol = pickle.HIGHEST_PROTOCOL)
                    
            with open(save_path + '/optical_flows_test_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as ofte:
                pickle.dump(self.optical_flows_test, ofte, protocol = pickle.HIGHEST_PROTOCOL)
                  
            with open(save_path + '/motion_representations_train_dt_{}.pkl'.format(int(self.delta_t)), 'wb') as mtr:
                pickle.dump(self.motion_representations_train, mtr, protocol = pickle.HIGHEST_PROTOCOL)
                
        data_logger.info('Success! Completed loading the data!')
 
       
        
                           