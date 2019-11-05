"""Show the usage of variables."""
import numpy as np
import tensorflow as tf


# Assign a value to the variable.
variable = tf.Variable(np.array([[0, 1], [2, 3]]))
print(variable)
"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int64, numpy=
array([[0, 1],
       [2, 3]])>
"""

# Assign a new value to the variable.
variable.assign(variable + np.ones((2, 2)))
print(variable)
"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int64, numpy=
array([[1, 2],
       [3, 4]])>
"""
