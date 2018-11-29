"""A simple script that produces some error message from tensorflow."""
import numpy as np
import tensorflow as tf


x = tf.placeholder(tf.float32, [None, 10])
x1 = tf.reshape(x, [-1, 1])
c = tf.matmul(x1 + x1, x + x)
with tf.Session() as session:
    session.run(c, feed_dict={x: np.random.randn(11, 10)})
