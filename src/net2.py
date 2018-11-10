import tensorflow as tf
import numpy as np
from dataset import *
import cv2
from optical_flow_computation import *
import pickle

def build_encoder_net(x, dt, ofp, is_training):
    c1 = tf.layers.conv2d(x, 32, (5,5), padding='SAME', activation=tf.nn.relu)
    p1 = tf.layers.conv2d(c1, 32, (2,2), strides=2, padding='VALID')

    c2 = tf.layers.conv2d(p1, 64, (5,5), padding='SAME', activation=tf.nn.relu)
    p2 = tf.layers.conv2d(c2, 64, (2,2), strides=2, padding='VALID')

    c3 = tf.layers.conv2d(p2, 128, (1,1), padding='SAME', activation=tf.nn.relu)
    p3 = tf.layers.conv2d(c3, 128, (2,2), strides=2, padding='VALID')

    flat = tf.layers.flatten(p3)

    fc1 = tf.layers.dense(flat, 7200)
    fc2 = tf.layers.dense(fc1, 4096)

    fct1 = tf.layers.dense(dt, 64)
    fct2 = tf.layers.dense(fct1, 64)
    fct3 = tf.layers.dense(fct2, 64)

    fcop = tf.layers.flatten(ofp)

    out = tf.keras.layers.concatenate([fc2, fct3, fcop], axis=1)
    return out


def build_decoder_net(x, is_training):
    fc1 = tf.layers.dense(x, 7200)
    fc2 = tf.layers.dense(fc1, 128*15*15)

    uc1 = tf.keras.layers.Reshape((15, 15, 128))(fc2)
    up1 = tf.layers.conv2d_transpose(uc1, 128, (2,2), strides=2, padding='VALID')

    uc2 = tf.layers.conv2d_transpose(up1, 64, (1,1), padding='SAME')
    up2 = tf.layers.conv2d_transpose(uc2, 64, (2,2), strides=2, padding='VALID')

    uc3 = tf.layers.conv2d_transpose(up2, 32, (5,5), padding='SAME')
    up3 = tf.layers.conv2d_transpose(uc3, 32, (2,2), strides=2, padding='VALID')

    uc4 = tf.layers.conv2d_transpose(up3, 1, (5,5), padding='SAME')
    return uc4

def train(num_epoch, lr, pt, ft, bs, print_every):
    tf.reset_default_graph()

    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, [None, 120, 120, 1])
        y = tf.placeholder(tf.float32, [None, 120, 120, 1])
        ofp = tf.placeholder(tf.float32, [None, 120, 120, 1])
        off = tf.placeholder(tf.float32, [None, 120, 120, 1])
        dt = tf.placeholder(tf.float32, [None, 1])
        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32)

        e = build_encoder_net(x, dt, ofp, is_training)
        out = build_decoder_net(e, is_training)

        loss_mat = tf.losses.mean_squared_error(labels = y, predictions = out) #, weights = off
        loss = tf.reduce_mean(loss_mat)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        ds = Dataset("../datasets/walking")
        x_t = None
        y_t = None
        ofp_t = None
        dt_t = np.full((1, 1), ft)
        for i in range(num_epoch):
            ds.set_values(bs, ft, pt)
            for t, X in enumerate(ds):
                x_np = np.array([X[index][pt] for index in range(len(X))])
                y_np = np.array([X[index][-1] for index in range(len(X))])
                ofp_np = np.array([Optical_Flow.compute(X[index][:pt+1]) for index in range(len(X))])
                off_np = np.array([Optical_Flow.compute(X[index][pt:]) for index in range(len(X))])
                dt_np = np.full((len(X), 1), ft)

                if t == 0:
                    x_t = x_np
                    y_t = y_np
                    ofp_t = ofp_np
                    dt_t = dt_np
                    continue

                feed_dict = {x:x_np, y:y_np, dt:dt_np, ofp:ofp_np, off:off_np, is_training:1, learning_rate:lr}
                loss_np,_ = sess.run([loss, train_op], feed_dict=feed_dict)

                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))

            # save_path = tf.train.Saver().save(sess, "../model/onet.ckpt")
            # print("Model saved at {}".format(save_path))
            print("COMPLETED EPOCH {}".format(i))
        
        with open("x_t", 'wb') as f:
            pickle.dump(x_t, f)
        with open("y_t", 'wb') as f:
            pickle.dump(y_t, f)
        with open("dt_t", 'wb') as f:
            pickle.dump(dt_t, f)

        feed_dict = {x:x_t, y:y_t, dt:dt_t, ofp:ofp_t, is_training:1}
        out_np = sess.run(out, feed_dict = feed_dict)

        for index in range(x_t.shape[0]):
            #print("Writing input test image")
            cv2.imwrite("in{}.jpg".format(index), x_t[index])

            out_np = out_np/(np.max(out_np, axis=(1,2,3))[:, None, None, None])
            out_np = out_np*255
            #print("Writing predicted output test")
            cv2.imwrite("out_pred{}.jpg".format(index), out_np[index].astype(dtype=np.uint8))

            #print("Writing groundtruth output test")
            cv2.imwrite("out_real{}.jpg".format(index), y_t[index])

if __name__ == '__main__':
    train(num_epoch=5, lr=5e-4, pt=5, ft=1, bs=16, print_every=15)
