#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:26:08 2018

@author: roshanprakash
"""

import tensorflow as tf


def baseline_pass(x, delta_t, optical_flows, is_training):
    
    """ Computes one forward pass of the data through the baseline encoder-decoder network.
        param x : the input images ; type --> numpy array of shape (N, 120, 120, 1)
        param delta_t : the difference in timesteps(interpreted as milliseconds) between 
                any input image and the desired target image; type --> float
        param optical_flows : the optical flow inputs corresponding to the image inputs.
                                type --> numpy array of shape (N, 120, 120, 1)
        returns : output images of shape (N, 120, 120, 1) resulting after a forward pass through the network.
    """
    
    # some model-specific setup
    activation_fn = tf.nn.relu
    initializer = tf.initializers.variance_scaling
    
                ############################   ENCODER   ############################
    
    ########## NOTE ---> Pooling layer is a 2*2 convolutional layer with filters = 1; no activation ##########
    
            ################## input image branch of the encoder #######################

    # pad the input before feeding to first convolutional layer
    padding_conv_1_im = tf.constant([[0, 0],[2, 2],[2, 2],[0, 0]])
    pad_x = tf.pad(x, paddings = padding_conv_1_im)
    
    # first convolutional layer
    conv_1_out_im = tf.layers.conv2d(pad_x, filters = 32, kernel_size = [5, 5], strides = [1, 1], \
                              padding = 'valid', activation = activation_fn, kernel_initializer = initializer)
    # shape of conv_1_out --> (32, 120, 120, 32)

    pool_1_out_im = tf.layers.conv2d(conv_1_out_im, filters = 32, kernel_size = [2, 2], strides = [2, 2], \
                                  padding = 'same', kernel_initializer = initializer)
    # shape of pool_1_out --> (32, 60, 60, 32)
    
    pool_1_out_im = tf.layers.dropout(pool_1_out_im, rate = 0.8, training = is_training) # dropout
    
    # pad the input before feeding to next layer
    padding_conv_2_im = tf.constant([[0, 0],[2, 2],[2, 2],[0, 0]])
    pool_1_out_im = tf.pad(pool_1_out_im, padding_conv_2_im)
    
    # second convolutional layer
    conv_2_out_im = tf.layers.conv2d(pool_1_out_im, filters = 64, kernel_size = [5, 5], strides = [1, 1], \
                                     padding = 'valid', activation = activation_fn, kernel_initializer = initializer)
    # shape of conv_2_out --> (32, 60, 60, 64)
    
    pool_2_out_im = tf.layers.conv2d(conv_2_out_im, filters = 64, kernel_size = [2, 2], strides = [2, 2], \
                                  padding = 'same', kernel_initializer = initializer)
    # shape of pool_2_out --> (32, 30, 30, 64)
    
    pool_2_out_im = tf.layers.dropout(pool_2_out_im, rate = 0.8, training = is_training) # dropout
    
    
    conv_3_out_im = tf.layers.conv2d(pool_2_out_im, filters = 128, kernel_size = [1, 1], strides = [1, 1], \
                                  padding = 'same', activation = activation_fn, kernel_initializer = initializer)
    
    pool_3_out_im = tf.layers.conv2d(conv_3_out_im, filters = 128, kernel_size = [2, 2], strides = [2, 2], \
                                  padding = 'same', kernel_initializer = initializer)
    # shape of pool_3_out --> (32, 15, 15, 128)
    
    pool_3_out_im = tf.layers.dropout(pool_3_out_im, rate = 0.8, training = is_training) # dropout
    
    # flatten input before feeding into FC layer
    pool_3_out_im = tf.layers.flatten(pool_3_out_im)

    fc_img_1_out = tf.layers.dense(pool_3_out_im, units = 7200, activation = activation_fn, \
                                    kernel_initializer = initializer)
    fc_img_1_out = tf.layers.dropout(fc_img_1_out, rate = 0.8, training = is_training) # dropout
    
    fc_img_2_out = tf.layers.dense(fc_img_1_out, units = 4096, activation = activation_fn, \
                                    kernel_initializer = initializer)
    

            #################### delta_t branch of the encoder #########################
                                    
    # delta_t is an (N, 1) vector with all same values
    fc_time_1_out = tf.layers.dense(delta_t, units = 64, activation = activation_fn, kernel_initializer = initializer)
    
    fc_time_1_out = tf.layers.dropout(fc_time_1_out, rate = 0.8, training = is_training) # dropout
    
    fc_time_2_out = tf.layers.dense(fc_time_1_out, units = 64, activation = activation_fn, kernel_initializer = initializer)
    
    fc_time_2_out = tf.layers.dropout(fc_time_2_out, rate = 0.8, training = is_training) # dropout
    
    fc_time_3_out = tf.layers.dense(fc_time_2_out, units = 64, activation = activation_fn, kernel_initializer = initializer)
           
    
                ############################   DECODER   ############################
    
    # concatenate the multi-input forward flow computed so far into a single input for the decoder model #
    decode_input = tf.concat([fc_img_2_out, fc_time_3_out], axis = 1)
    # shape of decode input --> (N, 4160)
    
    decode_input = tf.layers.dropout(decode_input, rate = 0.8, training = is_training) # dropout
    
    fc_decode_1_out = tf.layers.dense(decode_input, units = 7200, activation = activation_fn, \
                                    kernel_initializer = initializer)
    
    fc_decode_1_out = tf.layers.dropout(fc_decode_1_out, rate = 0.8, training = is_training) # dropout
    
    fc_decode_2_out = tf.layers.dense(fc_decode_1_out, units = 28800, activation = activation_fn, \
                                    kernel_initializer = initializer)
    
    fc_decode_2_out = tf.layers.dropout(fc_decode_2_out, rate = 0.8, training = is_training) # dropout
    
    # reshape to (N, 15, 15, 128) before actually decoding
    fc_decode_2_out = tf.reshape(fc_decode_2_out, (-1, 15, 15, 128))
    
    unpool_1_out = tf.layers.conv2d_transpose(fc_decode_2_out, filters = 128, kernel_size = [2, 2], strides = [2, 2], \
                                               padding = 'same', kernel_initializer = initializer)
    # shape of unpool_1_out --> (N, 30, 30, 128)
    
    deconv_1_out = tf.layers.conv2d_transpose(unpool_1_out, filters = 64, kernel_size = [1, 1], strides = [1, 1], \
                                              padding = 'valid', activation = activation_fn, kernel_initializer = initializer)
    # shape of deconv_1_out --> (N, 30, 30, 64)
    
    deconv_1_out = tf.layers.dropout(deconv_1_out, rate = 0.8, training = is_training) # dropout
    
    unpool_2_out = tf.layers.conv2d_transpose(deconv_1_out, filters = 64, kernel_size = [2, 2], strides = [2, 2], \
                                              padding = 'same', kernel_initializer = initializer)
    # shape of unpool_2_out --> (N, 60, 60, 64)

    deconv_2_out = tf.layers.conv2d_transpose(unpool_2_out, filters = 32, kernel_size = [1, 1], strides = [1, 1], \
                                              padding = 'valid', activation = activation_fn, kernel_initializer = initializer)
    # shape of deconv_2_out --> (N, 60, 60, 32)

    deconv_2_out = tf.layers.dropout(deconv_2_out, rate = 0.8, training = is_training) # dropout
    
    unpool_3_out = tf.layers.conv2d_transpose(deconv_2_out, filters = 32, kernel_size = [2, 2], strides = [2, 2], \
                                              padding = 'same', kernel_initializer = initializer)
    # shape of unpool_3_out --> (N, 120, 120, 32)
    
    model_out = tf.layers.conv2d_transpose(unpool_3_out, filters = 1, kernel_size = [1, 1], strides = [1, 1], \
                                              padding = 'valid', activation = activation_fn, 
                                              kernel_initializer = initializer, name='model_out')
    # shape of model_out --> (N, 120, 120, 1)

    return model_out