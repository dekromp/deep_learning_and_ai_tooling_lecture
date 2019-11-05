"""Shows the usage of the functional API of keras."""
import numpy as np
import tensorflow as tf


# Set the random seeds to make the outputs deterministic.
np.random.seed(1)
tf.random.set_seed(1)


# Generate some random data.
n = 10
x = np.random.randn(n, 3)
y = np.random.choice([0, 1], size=n)

# Build the layers as usual.
layer1 = tf.keras.layers.Dense(5, activation=tf.keras.activations.tanh)
layer2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

# Build up a keras model as a DAG. For this we need a symbolic representation
# of the input.
input_x = tf.keras.Input(shape=(3,))  # shape does not include the batch size.
# Than use it as an input to the layers.
h = layer1(input_x)
output = layer2(h)

# We wrap the inputs and outputs in a keras model.
model = tf.keras.Model(inputs=[input_x], outputs=[output])

# Now we can apply the model on the data.
p = model.predict(x)

print(p)
"""
[[0.32037497]
 [0.55872416]
 [0.32047206]
 [0.5434626 ]
 [0.536034  ]
 [0.5550912 ]
 [0.50310946]
 [0.48670274]
 [0.37343627]
 [0.5778259 ]]
"""
