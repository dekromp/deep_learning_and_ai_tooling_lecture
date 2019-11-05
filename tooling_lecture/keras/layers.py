"""Show the usage of keras layers."""
import numpy as np
import tensorflow as tf


# Building two layers with keras.
first_layer = tf.keras.layers.Dense(
    units=5, activation=tf.keras.activations.tanh
)
second_layer = tf.keras.layers.Dense(
    units=1, activation=tf.keras.activations.sigmoid
)


# Applying the layers on some input.
x = np.array([[1, 3, 4], [2, 2, 5]], dtype=np.float32)
print(x.shape)
"""
(2, 3)
"""
h = first_layer(x)
print(h.shape)
"""
(2, 5)
"""
output = second_layer(h)
print(output.shape)
"""
(2, 1)
"""


# Layers can be further configured.
third_layer = tf.keras.layers.Dense(
    units=10,
    activation=tf.keras.activations.relu,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.glorot_uniform(),
    bias_initializer=tf.keras.initializers.zeros(),
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    bias_regularizer=tf.keras.regularizers.l2(0.001),
)
