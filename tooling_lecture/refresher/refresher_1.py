"""Simple example script that shows basic operations of tensorflow."""
import numpy as np
import tensorflow as tf


# Fix the random seeds to make the computations reproducable.
tf.set_random_seed(12345)
np.random.seed(12321)

# Create an placeholder for feeding inputs in the graph.
input_x = tf.placeholder(tf.float32, [None, 3], name='features')

# Create a variable.
w = tf.get_variable(
    'weights', [3, 1], initializer=tf.glorot_uniform_initializer())

# Perform some computation steps.
output = tf.matmul(input_x, w)
output = tf.reshape(output, [-1])  # Flatten the outputs.

# Generate some random input data.
x = np.random.randn(5, 3)

# Execute the graph on some random data.
with tf.Session() as session:
    # Boilerplate code that initializes all variables in the graph (just w).
    session.run(tf.global_variables_initializer())
    output_value = session.run(output, feed_dict={input_x: x})
    print('Output: %s' % str(output_value))
    # Output: [ 1.382279  -0.9660325 -0.5551475  0.1781615 -1.5802894]
