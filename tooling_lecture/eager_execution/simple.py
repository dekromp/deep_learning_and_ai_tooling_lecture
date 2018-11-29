"""This script shows a simple example on how eager execution works."""
import numpy as np
import tensorflow as tf


# Enable eager execution.
tf.enable_eager_execution()

# Make the execution reproducable.
tf.set_random_seed(2132)
np.random.seed(3423)

# Generate some random data.
x = np.arange(3).reshape(-1, 1).astype(np.float32)
w = tf.get_variable(
    'w', dtype=np.float32, shape=[1, 3],
    initializer=tf.glorot_uniform_initializer())

# Interwine python and tensorflow code directly.
z = tf.matmul(w, x)
for i in range(5):
    if i % 2 == 0:
        h = -tf.nn.sigmoid(z)
    else:
        h = tf.nn.sigmoid(z)

    # Evaluate immediately the output without session run.
    print(h)

# tf.Tensor([[-0.36252844]], shape=(1, 1), dtype=float32)
# tf.Tensor([[0.36252844]], shape=(1, 1), dtype=float32)
# tf.Tensor([[-0.36252844]], shape=(1, 1), dtype=float32)
# tf.Tensor([[0.36252844]], shape=(1, 1), dtype=float32)
# tf.Tensor([[-0.36252844]], shape=(1, 1), dtype=float32)
