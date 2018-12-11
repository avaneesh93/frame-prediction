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

    with tf.device('/device:GPU:0'):
        
        # call function to compute forward pas
        # init = tf.global_variables_initializer()
        

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        try:
            # sess.run(init)
            saver = tf.train.import_meta_graph(restore_path + '.meta')
            saver.restore(sess, restore_path)
            print('Loading the trained model saved previously.')

            graph = tf.get_default_graph()
            is_training = graph.get_tensor_by_name("is_training:0")
            X = graph.get_tensor_by_name("X_batch:0")
            y = graph.get_tensor_by_name("y_batch:0")
            delta_t = graph.get_tensor_by_name("delta_t_batch:0")
            optical_flows = graph.get_tensor_by_name("optical_flows_batch:0")
            motion_representations = graph.get_tensor_by_name("motion_representations_batch:0")
            loss = graph.get_tensor_by_name("loss:0")
            
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
