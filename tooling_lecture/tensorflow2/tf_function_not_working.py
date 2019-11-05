"""Shows limits of tf.function."""
import tensorflow as tf


@tf.function
def inplace_operation(x):
    """Do something.

    Parameters
    ----------
    x : list
        A list input.

    Returns
    -------
    list
        The updated list.

    """
    x.append(4)
    return x


# Raises exeption.
# print(inplace_operation([8]))


import numpy as np

x = tf.constant(np.random.randn(3, 4))
x[1, :] = 3
