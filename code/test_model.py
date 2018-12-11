#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:20:04 2018

@author: roshanprakash
"""

import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import pickle
from dataset_loader import *
from encode_decode import *

import logging
logging.basicConfig(level = logging.INFO)
model_logger = logging.getLogger(__name__ + '.model_tester')


def test_model(X_test, y_test, dt, optical_flows_test):
    
    """                                 MODEL TESTING PROCEDURE
    
        param X_test : the test images of shape (N', 120, 120, 1); to be loaded from memory before passing in ; type --> numpy array
        param y_test : the test-time targets of shape (N', 120, 120, 1); to be loaded from memory before passing in ; type --> numpy array
        param dt : a scalar indicating the difference in timesteps(interpreted as milliseconds) between 
                        any input image and the desired target image; type --> float 
        param optical_flows_test : the optical flow inputs for corresponding test images, of shape (N', 120, 120, 1) ; 
                            to be loaded from memory before passing in ; type --> numpy array
        returns : None; saves all predictions in numerical and image formats to results directory.
    """
    
    save_path = os.path.dirname(os.getcwd()) + '/results'
    restore_path = os.path.dirname(os.getcwd()) + '/model/model.ckpt' 
    tf.reset_default_graph()
    
    with tf.device('/device:GPU:0'):
        
        # initialize tf variables
        is_training = tf.placeholder(tf.bool, name='is_training')
        X = tf.placeholder(tf.float32, [None, 120, 120, 1])
        delta_t = tf.placeholder(tf.float32, [None, 1])
        optical_flows = tf.placeholder(tf.float32, [None, 120, 120, 1])
        
        # call function to compute forward pass
        model_out = encoder_decoder_pass(X, delta_t, optical_flows, is_training) * 255.0
        saver = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        try:
            saver.restore(sess, restore_path)
            model_logger.info('Loading the trained model saved previously.')
            
        except:
            raise ValueError('No trained model found. Train a model and then try again!')
            
        model_logger.info('Checking model performance on test data now. Process starting.')
        
        test_size = X_test.shape[0]

        X_test_batches = np.split(X_test, np.arange(16, test_size, 16))
        optical_flows_test_batches = np.split(optical_flows_test, np.arange(16, test_size, 16))
        test_predictions = None

        model_logger.info('Total number of batches to test = {}'.format(len(X_test_batches)))

        for batch_index in range(len(X_test_batches)):
            if (batch_index + 1)%10:
                model_logger.info('Testing batch {}'.format(batch_index+1))
            feed = {X: X_test_batches[batch_index], delta_t : np.ones((X_test_batches[batch_index].shape[0], 1)) * dt, 
                        optical_flows : optical_flows_test_batches[batch_index], is_training : False}
            new_test_predictions = sess.run(model_out, feed_dict = feed) * 255.0 # these are normalized outputs. Transform back.
            if test_predictions is None:
                test_predictions = new_test_predictions
            else:
                test_predictions = np.concatenate((test_predictions, new_test_predictions))
        
        # save predictions to avoid re-run
        # with open(save_path + '/test_predictions.pkl', 'wb') as t:
        #     pickle.dump(test_predictions, t) 
        # model_logger.info('Dumped test predictions.')
 
        # convert predictions to images and save them for analysis
        for img_idx in range(test_predictions.shape[0]):
            cv.imwrite(os.path.join(save_path, 'inputs/input_{}.jpg'.format(img_idx + 1)), X_test[img_idx])
            cv.imwrite(os.path.join(save_path, 'predictions/prediction_{}.jpg'.format(img_idx + 1)), test_predictions[img_idx])
            cv.imwrite(os.path.join(save_path, 'targets/target_{}.jpg'.format(img_idx + 1)), y_test[img_idx] * 255.0)
        
        model_logger.info('Saved all input, predicted and target images(.jpg format) to results directory.')
    
    model_logger.info('Testing complete.')
    
if __name__ == '__main__':
      
    data = dataset_loader(delta_t = 40.0, k = 10, offset = 10) # make sure pickled data is loaded. should use data identical/comparable to training data. 
    
    model_logger.info('Starting testing procedure now.')
    test_model(data.X_test['walking'], data.y_test['walking'], data.delta_t, data.optical_flows_test['walking'])
