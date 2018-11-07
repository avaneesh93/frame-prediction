import tensorflow as tf

def build_encoder_net(x, dt, is_training):
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
    fct3 = tf.layers.dense(fct3, 64)

    out = tf.keras.layers.concatenate([fc2, fct3], axis=1)
    return out


def build_decoder_net(x, is_training):
    pass


if __name__ == '__main__':
    pass
