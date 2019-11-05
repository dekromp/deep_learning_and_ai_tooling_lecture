"""Shows how customizable keras is."""
import numpy as np
import tensorflow as tf


# Parameters initializers.
initializer = tf.keras.initializers.constant(np.random.randn(1, 10))


def my_regularizer(weight_matrix):
    """A custom regularizer."""
    return 0.01 * tf.reduce_sum(weight_matrix)


# Layers can be further configured.
layer = tf.keras.layers.Dense(
    units=10,
    activation=tf.keras.activations.relu,
    kernel_initializer=initializer,
    kernel_regularizer=my_regularizer,
)
