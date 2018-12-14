#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:03:19 2018

@author: roshanprakash
"""

import os
import numpy as np
import tensorflow as tf
from dataset_loader import *
from encode_decode import *

import logging
logging.basicConfig(level = logging.INFO)
model_logger = logging.getLogger(__name__ + '.model_trainer')


def train_model(X_train, y_train, delta_t, optical_flows_train, motion_representations_train, lr, epochs, tune, print_every = 1):
    
    """                                 MODEL TRAINING PROCEDURE
    
        param X_train : the training images of shape (N, 120, 120, 1); to be loaded from memory before passing in ; type --> numpy array
        param y_train : the target images of shape (N, 120, 120, 1); to be loaded from memory before passing in ; type --> numpy array
        param delta_t : a scalar indicating the difference in timesteps(interpreted as milliseconds) between 
                        any input image and the desired target image; type --> float 
            WARNING : Before proceeding, double-check if this delta_t value is consistent with the 'delta_t' used with the dataset object!
        param optical_flows_train : the optical flow inputs for corresponding training images, of shape (N, 120, 120, 1) ; 
                                to be loaded from memory before passing in ; type --> numpy array  
        param motion_representations_train : the motion representation inputs for corresponding training images, of shape (N, 120, 120, 1) ; 
                                to be loaded from memory before passing in ; type --> numpy array    
        
        --------------------  FOR MORE DETAILS ON THESE INPUT PARAMETERS, LOOK UP THE DATASET CLASS DESCRIPTIONS! --------------------------
        
        param lr : the learning rate for weight updates ; type --> float
        param epochs : the number of training epochs ; type --> int
        param tune : a boolean indicating whether this train function is called during a hyperparameter tuning procedure
        param print_every : prints training stats after each set of this number of epochs; type --> int
    """     
    
    save_path = os.path.dirname(os.getcwd()) + '/model'   
    tf.reset_default_graph()
    
    with tf.device('/device:GPU:0'):
        
        # initialize tf variables
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        X_batch = tf.placeholder(tf.float32, shape=[None, 120, 120, 1], name='X_batch')
        y_batch = tf.placeholder(tf.float32, shape=[None, 120, 120, 1], name='y_batch')
        delta_t_batch = tf.placeholder(tf.float32, shape=[None, 1], name='delta_t_batch')
        optical_flows_batch = tf.placeholder(tf.float32, shape=[None, 120, 120, 1], name='optical_flows_batch')
        motion_representations_batch = tf.placeholder(tf.float32, shape=[None, 120, 120, 1], name='motion_representations_batch')
        
        # call function to compute forward pass
        model_out = encoder_decoder_pass(X_batch, delta_t_batch, optical_flows_batch, is_training)
        
        # loss computation
        losses = tf.losses.mean_squared_error(labels = y_batch * 255.0, predictions = model_out * 255.0)
        losses *= (motion_representations_batch) 
        loss = tf.reduce_mean(losses, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        # weight updates
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss)
        
        saver = tf.train.Saver()
        model_logger.info('Completed setting up the computational graph!')
        
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        if tune:
            all_losses = []
          
        for i in range(1, epochs+1):
            
            # first, call the get_batch() function to gather batches of training data
            X_, y_, OF_, MR_ = get_batches(X_train, y_train, optical_flows_train, motion_representations_train)
            
            for j in range(X_.shape[0]):
 
                # setup the feed dictionary (avoid naming conflicts!)
                feed = {X_batch : X_[j], y_batch : y_[j], delta_t_batch : np.ones((16, 1)) * delta_t, 
                            optical_flows_batch : OF_[j], motion_representations_batch: MR_[j], is_training : True, learning_rate : lr}
                    
                batch_loss, _ =  sess.run([loss, train_step], feed_dict = feed)
                # model_logger.info('Completed iteration {}'.format(i * j))
   
            if tune:
                all_losses.append(batch_loss)    
            if i % print_every == 0:
                model_logger.info('Completed epoch {}'.format(i))    
                model_logger.info('Loss after epoch {} : {}'.format(i, batch_loss))
      
        model_logger.info("All epochs of training now complete.")
        
        if not tune:
            saver.save(sess, save_path + '/model.ckpt') # save the trained model
            model_logger.info('Saved the model to results directory.')
        
        model_logger.info('Training complete.')
        
        if tune:
            return all_losses
        
        
if __name__ == '__main__':
      
    data = dataset_loader(delta_t = 40.0, k = 10, offset = 10)  
    model_logger.info('Starting training procedure now.')
    train_model(data.X_train['walking'], data.y_train['walking'], data.delta_t, data.optical_flows_train['walking'], data.motion_representations_train['walking'], lr = 1e-5, epochs = 150, tune = False, print_every = 1)
