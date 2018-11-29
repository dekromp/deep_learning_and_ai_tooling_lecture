import numpy as np
from numpy.testing import assert_almost_equal
import tensorflow as tf

from .tensorflow_example import sigmoid


def test_sigmoid():
    x = np.arange(-100, 100, 10).astype(np.float32)
    x_input = tf.placeholder(tf.float32, [None])

    with tf.Session() as session:
        r1, r2 = session.run(
            [tf.sigmoid(x), sigmoid(x)], feed_dict={x_input: x})
        assert_almost_equal(r1, r2)

        x[0] = 88.38223
        print(session.run(tf.exp(x_input), feed_dict={x_input: x}))
        print(session.run(tf.sigmoid(x_input), feed_dict={x_input: x}))
        print(session.run(sigmoid(x_input), feed_dict={x_input: x}))


def test_binary_cross_entropy():
    logits = tf.placeholder(tf.float32, [None])
    input_y = tf.placeholder(tf.float32, [None])

    output = tf.nn.sigmoid(logits)

    epsilon = 1e-7
    loss = -(tf.multiply(input_y, tf.log(output + epsilon)) +
             tf.multiply(1 - input_y, tf.log(1 - output + epsilon)))
    loss = tf.reduce_mean(loss, name='loss_out')

    loss2 = tf.losses.log_loss(input_y, output)
    loss3 = tf.losses.sigmoid_cross_entropy(input_y, logits)

    with tf.Session() as session:
        size = 100
        labels = np.random.choice([0, 1], size=size).astype(np.float32)
        values = np.random.uniform(-1, 1, size=size).astype(np.float32)

        l1, l2, l3 = session.run(
            [loss, loss2, loss3], feed_dict={input_y: labels, logits: values})

        assert_almost_equal(l1, l2)
        assert_almost_equal(l1, l3, decimal=5)
