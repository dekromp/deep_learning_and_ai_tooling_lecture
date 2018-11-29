"""Example for nice code."""
import numpy as np
from numpy.testing import assert_array_almost_equal

from tooling_lecture.code_style.documented_function_example import fit_linear


np.random.seed(123456)


def test_fit_linear():
    """The the fit_linear function from the slides."""
    # Generate a random linear regression model on random data.
    x = np.random.randn(100, 3)
    true_w = np.array([0.3, -0.21, 0.8])
    y = np.dot(x, true_w)

    # Use our function to compute the parameters.
    w = fit_linear(x, y)

    # Should be the same as the true w.
    assert_array_almost_equal(true_w, w)
