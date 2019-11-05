"""Shows the usage of tf.function."""
import numpy as np
import tensorflow as tf


@tf.function
def absolute_value(x):
    """Compute the absolute value.

    Parameters
    ----------
    x : float

    Returns
    -------
    float
        The absolute value.

    """
    if x > 0:
        return x
    else:
        return -x


print(absolute_value)
"""Prints:
<tensorflow.python.eager.def_function.Function object at 0x7f89eb2ea3c8>
"""

print(absolute_value(4))
"""Prints:
tf.Tensor(4, shape=(), dtype=int32)
"""

print(absolute_value(-99))
"""Prints:
tf.Tensor(99, shape=(), dtype=int32)
"""


@tf.function
def looping(x):
    """Using a for loop do something useful.

    Parameters
    ----------
    x : numpy.ndarray
        The input.

    Returns
    -------
    list
        The results.

    """
    output = []
    for i in range(3):
        output.append(x + i)

    return output


print(looping(np.zeros((2, 1))))
"""
[<tf.Tensor: id=27, shape=(2, 1), dtype=float64, numpy=
array([[0.],
       [0.]])>, <tf.Tensor: id=28, shape=(2, 1), dtype=float64, numpy=
array([[1.],
       [1.]])>, <tf.Tensor: id=29, shape=(2, 1), dtype=float64, numpy=
array([[2.],
       [2.]])>]
"""
