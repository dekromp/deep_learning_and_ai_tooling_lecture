"""Shows the numpy integration of tensorflow."""
import numpy as np
import tensorflow as tf


# Create a 3 x 4 array of float values.
x = np.array([[1, 4, 4, 3], [2, 3, 6, 0], [9, 3, 2, 8]], dtype=np.float32)

print(x)
"""
array([[1., 4., 4., 3.],
       [2., 3., 6., 0.],
       [9., 3., 2., 8.]], dtype=float32)
"""

# Tensorflow 2.0 operations accept numpy arrays as inputs.
z = tf.matmul(x, x.T)
print(x)
"""
<tf.Tensor: id=5, shape=(3, 3), dtype=float32,
 numpy=array([[ 42.,  38.,  53.],
              [ 38.,  49.,  39.],
              [ 53.,  39., 158.]], dtype=float32)>
"""
