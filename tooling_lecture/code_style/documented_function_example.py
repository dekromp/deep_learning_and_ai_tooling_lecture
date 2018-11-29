"""Example for bad and nice code."""
import numpy as np


def f(x, y):
    xtxi = np.linalg.pinv(np.dot(x.T, x))
    xty = np.dot(x.T, y)
    w = np.dot(xtxi, xty)
    return w


def fit_linear(x, y):
    """Compute the parameters of a linear regression model in closed form.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        The feature data.
    y : :class:`numpy.ndarray`
        The target data.

    Returns
    -------
    w : :class:`numpy.ndarray`
        The parameters of the linear regression model.

    """
    # Compute the pseudo-inverse of the covariance matrix.
    xtxi = np.linalg.pinv(np.dot(x.T, x))

    # Compute the parameters of the linear model using the closed form solution
    # w = (XtX)^-1 * Xt * y
    xty = np.dot(x.T, y)
    w = np.dot(xtxi, xty)

    return w
