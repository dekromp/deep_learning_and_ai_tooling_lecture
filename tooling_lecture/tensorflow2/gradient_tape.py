"""Show the usage of the gradient tape."""
import numpy as np
import tensorflow as tf

np.random.seed(1)  # For reproducibility.


def generate_data():
    """Generate some random dummy data that is easy to fit.

    Returns
    -------
    numpy.ndarray
        The feature data.
    numpy.ndarray
        The target data.

    """
    input_data = np.random.randn(100, 3)
    target_data = input_data.dot(np.random.uniform(1, 2, size=(3, 1))) + 2.1
    return input_data, target_data


def mean_absolute_error(y, estimate):
    """Compute the mean absolute error.

    Parameters
    ----------
    y : tf.Tensor
        The true values.
    estimate : tf.Tensor
        The output of the model

    """
    return tf.reduce_mean(tf.abs(y - estimate))


class MyModel(object):
    """A simple linear regression model."""

    def __init__(self):  # noqa
        self.weights = tf.Variable(np.random.randn(3, 1))
        self.bias = tf.Variable(np.array([0.0]))

    def forward_pass(self, x):
        """The forward computation of our model.

        y = wx +b

        Parameters
        ----------
        x : tf.Tensor
            The input to the model.

        Returns
        -------
        tf.Tensor
            The output of the model.

        """
        return tf.matmul(x, self.weights) + self.bias


def main():
    """Execute model training."""
    # Generate some data.
    input_data, target_data = generate_data()

    # Initialize the model.
    model = MyModel()

    # Compute the model output.
    model_output = model.forward_pass(input_data)
    print(mean_absolute_error(target_data, model_output))
    """
    tf.Tensor(3.3313408998663987, shape=(), dtype=float64)
    """

    learning_rate = 0.25
    for iteration in range(20):  # Train for 20 iterations.
        with tf.GradientTape() as tape:
            # The variables in out model are watched automatically.
            # Compute the outputs of the model.
            estimates = model.forward_pass(input_data)
            # Measure the error
            error = mean_absolute_error(target_data, estimates)
            print('Iteration %d model error: %.4f' % (iteration, error))

        # Compute the derivative of the error with respect to the parameters.
        dweights, dbias = tape.gradient(error, [model.weights, model.bias])

        # Vanilla gradient descent update rule.
        model.weights.assign(model.weights - learning_rate * dweights)
        model.bias.assign(model.bias - learning_rate * dbias)

    """
    Iteration 0 model error: 3.3313
    Iteration 1 model error: 3.1953
    Iteration 2 model error: 3.0596
    Iteration 3 model error: 2.9239
    Iteration 4 model error: 2.7882
    Iteration 5 model error: 2.6526
    Iteration 6 model error: 2.5169
    Iteration 7 model error: 2.3813
    Iteration 8 model error: 2.2456
    Iteration 9 model error: 2.1099
    Iteration 10 model error: 1.9743
    Iteration 11 model error: 1.8387
    Iteration 12 model error: 1.7037
    Iteration 13 model error: 1.5695
    Iteration 14 model error: 1.4352
    Iteration 15 model error: 1.3010
    Iteration 16 model error: 1.1668
    Iteration 17 model error: 1.0330
    Iteration 18 model error: 0.8996
    Iteration 19 model error: 0.7662
    """


if __name__ == '__main__':
    main()
