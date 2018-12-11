import os
import cv2 as cv
import tensorflow as tf
import numpy as np
from dataset_loader import *
from encode_decode import *
from helpers import *

def test_save(X_train, y_train, dt, optical_flows_train, motion_representations_train):

    save_path = os.path.dirname(os.getcwd()) + '/results'
    restore_path = os.path.dirname(os.getcwd()) + '/model/model.ckpt' 
    tf.reset_default_graph()

    with tf.device('/device:GPU:0'):
        
        # initialize tf variables
        is_training = tf.placeholder(tf.bool, name='is_training')
        X = tf.placeholder(tf.float32, [None, 120, 120, 1])
        y = tf.placeholder(tf.float32, [None, 120, 120, 1])
        delta_t = tf.placeholder(tf.float32, [None, 1])
        optical_flows = tf.placeholder(tf.float32, [None, 120, 120, 1])
        motion_representations = tf.placeholder(tf.float32, [None, 120, 120, 1])
        
        # call function to compute forward pass
        model_out = encoder_decoder_pass(X, delta_t, optical_flows, is_training)
        losses = tf.losses.mean_squared_error(labels = y * 255.0, predictions = model_out)
        # losses *= (motion_representations_batch) 
        loss = tf.reduce_mean(losses)
        saver = tf.train.import_meta_graph(restore_path + '.meta')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        try:
            saver.restore(sess, restore_path)
            print('Loading the trained model saved previously.')
            
        except:
            raise ValueError('No trained model found. Train a model and then try again!')

        X_, y_, OF_, MR_ = get_batches(X_train, y_train, optical_flows_train, motion_representations_train)
            
        feed = {X: X_[0], delta_t : np.ones((16, 1)) * dt, 
                        optical_flows : OF_[0], is_training : True, y: y_[0], motion_representations: MR_[0]}
        loss_ = sess.run(loss, feed_dict=feed)

        print(loss_)

if __name__ == '__main__':
      
    data = dataset_loader(delta_t = 40.0, k = 10, offset = 10) # make sure pickled data is loaded. should use data identical/comparable to training data. 
    
    print('Starting testing procedure now.')
    test_save(data.X_train['walking'], data.y_train['walking'], data.delta_t, data.optical_flows_train['walking'], data.motion_representations_train['walking'])
