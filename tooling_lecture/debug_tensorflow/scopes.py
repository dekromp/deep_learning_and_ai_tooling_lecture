"""A simple script that shows how scopes work."""
import numpy as np
import tensorflow as tf


x = tf.placeholder(tf.float32, [None, 10])

print('No scope is used:')
w1 = tf.get_variable(
    'v1', dtype=np.float32, initializer=np.ones((10, 3), dtype=np.float32))
h1 = tf.matmul(x, w1)

w2 = tf.get_variable(
    'v2', dtype=np.float32, initializer=np.ones((3, 10), dtype=np.float32))
h2 = tf.matmul(h1, w2)

for tensor in [x, w1, w2, h1, h2]:
    print(tensor)

print('\nScope is used:')
with tf.variable_scope('first_block'):
    w1 = tf.get_variable(
        'v1', dtype=np.float32, initializer=np.ones((10, 3), dtype=np.float32))
    h1 = tf.matmul(x, w1)

with tf.variable_scope('second_block'):
    w2 = tf.get_variable(
        'v2', dtype=np.float32, initializer=np.ones((3, 10), dtype=np.float32))
    h2 = tf.matmul(h1, w2)

for tensor in [x, w1, w2, h1, h2]:
    print(tensor)
