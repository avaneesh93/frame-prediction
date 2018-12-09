#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:37:56 2018

@author: roshanprakash
"""
import sys
import os
import cv2 as cv
import math
import numpy as np
import pickle 
from optical_flow import *
from helpers import *

import logging
logging.basicConfig(level = logging.INFO)
data_logger = logging.getLogger(__name__ + '.data')


class dataset_loader():
    
    def __init__(self, delta_t, k, offset, pickle_batch_size = 150):

        """ *arg data_dir : the source directory for the data, as a pathname ; type --> str
            
            *arg delta_t  : the difference in timesteps(interpreted as milliseconds) between 
                            any input image and the desired target image; type --> float
            
            *arg k : the number of previous(sequential) images to be used while computing the optical flow for any input image.
                        (should be fixed as a constant with this initialization) ; type --> int
            
            *arg offset : a value indicating the number of images to skip from every image sequence directory.
                        i.e, for every 'n' images in any sequence directory, images(1,..k) and images(n-k,...n) 
                        will be skipped. (to ignore images which do not contain potential motion information)   

            *arg pickle_batch_size : Number of sequences after the processing of which we attempt to save the sets to disk
        """
        
        self.path =  os.path.dirname(os.getcwd()) + '/datasets' # --> root directory + '/datasets'
        #self.path = '../datasets' while running from a script
        self.delta_t = delta_t
        self.future_frame_number = math.ceil((self.delta_t / 1000.0) * 24)
        self.k = k
        self.offset = offset
        self.means = {}
        self.stdevs = {}
        self.pickle_batch_size = pickle_batch_size
        self.load_data()
        
    
    def load_data(self):
        
        """ Loads the images from the data source and generates training and testing datasets."""
        
        load_path = os.path.dirname(os.getcwd()) + '/pickles'
        #load_path = '../pickles' while running from a script
        X_train_path = load_path + '/K_X_train_dt_{}'.format(int(self.delta_t))
        y_train_path = load_path + '/K_y_train_dt_{}'.format(int(self.delta_t))
        X_test_path = load_path + '/K_X_test_dt_{}'.format(int(self.delta_t))
        y_test_path = load_path + '/K_y_test_dt_{}'.format(int(self.delta_t))
        optical_flows_train_path = load_path + '/K_optical_flows_train_dt_{}'.format(int(self.delta_t))
        optical_flows_test_path = load_path + '/K_optical_flows_test_dt_{}'.format(int(self.delta_t))
        motion_representations_train_path = load_path + '/K_motion_representations_train_dt_{}'.format(int(self.delta_t))
        
        # First, lookup if pickled data is available for loading
        if os.path.exists(X_train_path) and os.path.exists(y_train_path) and os.path.exists(X_test_path) and  os.path.exists(y_test_path) \
                and os.path.exists(optical_flows_train_path) and  os.path.exists(optical_flows_test_path) and  os.path.exists(motion_representations_train_path):
            
            data_logger.info('Pickled data available. Loading now!')
            
            #data_logger.info("Custom loading. Pickle loading couldn't handle this huge data size!")
            # self.X_train = try_to_load_as_pickled_object_or_None(X_train_path)
            # self.y_train = try_to_load_as_pickled_object_or_None(y_train_path)
            # self.X_test = try_to_load_as_pickled_object_or_None(X_test_path)
            # self.y_test = try_to_load_as_pickled_object_or_None(y_test_path)
            # self.optical_flows_train = try_to_load_as_pickled_object_or_None(optical_flows_train_path)
            # self.optical_flows_test = try_to_load_as_pickled_object_or_None(optical_flows_test_path)
            # self.motion_representations_train = try_to_load_as_pickled_object_or_None(motion_representations_train_path)

            self.X_train = try_to_load_as_pickled_object_or_None(load_path, 'X_train_dt_{}'.format(int(self.delta_t)))
            self.y_train = try_to_load_as_pickled_object_or_None(load_path, 'y_train_dt_{}'.format(int(self.delta_t)))
            self.X_test = try_to_load_as_pickled_object_or_None(load_path, 'X_test_dt_{}'.format(int(self.delta_t)))
            self.y_test = try_to_load_as_pickled_object_or_None(load_path, 'y_test_dt_{}'.format(int(self.delta_t)))
            self.optical_flows_train = try_to_load_as_pickled_object_or_None(load_path, 'optical_flows_train_dt_{}'.format(int(self.delta_t)))
            self.optical_flows_test = try_to_load_as_pickled_object_or_None(load_path, 'optical_flows_test_dt_{}'.format(int(self.delta_t)))
            self.motion_representations_train = try_to_load_as_pickled_object_or_None(load_path, 'motion_representations_train_dt_{}'.format(int(self.delta_t)))
            
                
                
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
                    X, y, optical_flows, motion_representations = self.init_dicts(action_dir_name)
                    
                    action_path = os.path.join(self.path, action_dir_name)
                    
                    count = 1
                    # for every sequence directory within the action directory
                    for seq_name in os.listdir(action_path):
                        
                        print(count)
                        count+=1
                        images = [] # reset for every sequence of images since we will be gathering all inputs in a per-sequence directory manner
                        
                        if seq_name == '.DS_Store':
                            pass
                        
                        else:
                            seq_path = os.path.join(action_path, seq_name)
                            for img_name in sorted(os.listdir(seq_path), key=lambda filename: int(filename[:-4])):
                                images.append(cv.imread(os.path.join(seq_path, img_name)))
        
                            seq_size = len(images)
                            
                            if not seq_size > self.k + self.future_frame_number + (2 * self.offset):
                                data_logger.info('Not enough images in this sequence directory to build training and test images. So, ignoring this sequence directory for now!')
                            
                            else: # we have enough images to build model training and test data
                                
                                for img_idx in range(self.k + self.offset, seq_size - self.offset):
                                    # note here that we will need images from kth frame onward for optical flow inputs to be available as well !
                                    # also note that we will only require images for which targets(img[current_time_step + delta_t]) exist !
                                    if img_idx + self.future_frame_number < seq_size - self.offset:
                                        X[action_dir_name].append(images[img_idx])
                                        y[action_dir_name].append(images[img_idx + int(self.future_frame_number)])
                                        # (below) calculates and gathers the optical flow input for this image frame using 'k' previous image frames
                                        optical_flows[action_dir_name].append(compute_optical_flow(np.array(images[img_idx - self.k : img_idx + 1]))) 
                                        # (below) calculates and gathers the 'motion representation' for this image frame using all sequential future image frames until target image
                                        motion_representations[action_dir_name].append(compute_optical_flow(np.array(images[img_idx: img_idx + int(self.future_frame_number)+ 1])))
                    
                                data_logger.info('Sequence loading complete!')

                            if (count-1) % self.pickle_batch_size == 0:
                                data_logger.info('Writing training and test sets to disk')
                                self.archive(X, y, optical_flows, motion_representations, action_dir_name, load_path)
                                X, y, optical_flows, motion_representations = self.init_dicts(action_dir_name)
                    if len(X[action_dir_name]) > 0:
                        data_logger.info('Writing remaining training and test sets to disk')
                        self.archive(X, y, optical_flows, motion_representations, action_dir_name, load_path)
                
        data_logger.info('Success! Completed loading the data!')

    def archive(self, X, y, optical_flows, motion_representations, action_dir_name, path):
        X[action_dir_name] = np.asarray(X[action_dir_name])
        y[action_dir_name] = np.asarray(y[action_dir_name])
        optical_flows[action_dir_name] = np.asarray(optical_flows[action_dir_name])
        motion_representations[action_dir_name] = np.asarray(motion_representations[action_dir_name])

        # normalize and split

        N, H, W, C = X[action_dir_name].shape
                    
        X[action_dir_name] = X[action_dir_name] / 255.0
        y[action_dir_name] = y[action_dir_name] / 255.0
        optical_flows[action_dir_name] = optical_flows[action_dir_name]# already normalized
        motion_representations[action_dir_name] = motion_representations[action_dir_name]#[idxs][0] # already normalized

        # create training and testing datasets with batches for training data
        train_size = math.floor(0.8 * N)
        fixed_N = int((math.floor(train_size / 16.0) / (train_size / 16.0)) * train_size)

        new_X_train = X[action_dir_name][:fixed_N][: , : , :, [0]]
        new_y_train = y[action_dir_name][:fixed_N][: , : , :, [0]]
        new_X_test = X[action_dir_name][fixed_N:][: , : , :, [0]]
        new_y_test = y[action_dir_name][fixed_N:][: , : , :, [0]]
        new_optical_flows_train = optical_flows[action_dir_name][:fixed_N]
        new_optical_flows_test = optical_flows[action_dir_name][fixed_N:]
        new_motion_representations_train = motion_representations[action_dir_name][:fixed_N]

        if action_dir_name in self.X_train:
            self.X_train[action_dir_name] = np.concatenate((self.X_train[action_dir_name], new_X_train))
            self.y_train[action_dir_name] = np.concatenate((self.y_train[action_dir_name], new_y_train))
            self.X_test[action_dir_name] = np.concatenate((self.X_test[action_dir_name], new_X_test))
            self.y_test[action_dir_name] = np.concatenate((self.y_test[action_dir_name], new_y_test))
            self.optical_flows_train[action_dir_name] = np.concatenate((self.optical_flows_train[action_dir_name], new_optical_flows_train))
            self.optical_flows_test[action_dir_name] = np.concatenate((self.optical_flows_test[action_dir_name], new_optical_flows_test))
            self.motion_representations_train[action_dir_name] = np.concatenate((self.motion_representations_train[action_dir_name], new_motion_representations_train))
        else:
            self.X_train[action_dir_name] = new_X_train
            self.y_train[action_dir_name] = new_y_train
            self.X_test[action_dir_name] = new_X_test
            self.y_test[action_dir_name] = new_y_test
            self.optical_flows_train[action_dir_name] = new_optical_flows_train
            self.optical_flows_test[action_dir_name] = new_optical_flows_test
            self.motion_representations_train[action_dir_name] = new_motion_representations_train

        save_as_pickled_object(new_X_train, path, 'X_train_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_y_train, path, 'y_train_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_X_test, path, 'X_test_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_y_test, path, 'y_test_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_optical_flows_train, path, 'optical_flows_train_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_optical_flows_test, path, 'optical_flows_test_dt_{}'.format(int(self.delta_t)), action_dir_name)
        save_as_pickled_object(new_motion_representations_train, path, 'motion_representations_train_dt_{}'.format(int(self.delta_t)), action_dir_name)

    def init_dicts(self, action_dir_name):
        # Initializing batches
        X = {}
        y = {}
        optical_flows = {}
        motion_representations = {}
        
        X[action_dir_name] = []
        y[action_dir_name] = []
        optical_flows[action_dir_name] = []
        motion_representations[action_dir_name] = []

        return X, y, optical_flows, motion_representations

