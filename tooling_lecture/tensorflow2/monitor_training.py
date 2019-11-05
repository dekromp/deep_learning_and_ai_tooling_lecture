"""Show the usage of the gradient tape."""
import os
import shutil

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
    input_data = np.random.randn(1000, 30)
    target_data = input_data.dot(np.random.uniform(1, 2, size=(30, 1))) + 2.1
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
        self.weights = tf.Variable(np.random.uniform(0, 1, size=(30, 1)))
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
    """Train the model and write the summaries."""
    # Generate some data.
    input_data, target_data = generate_data()

    # Initialize the model.
    model = MyModel()

    # Define a writer that writes files that Tensorboard understand.
    this_dir = os.path.dirname(__file__)  # this dir.
    summary_dir = os.path.join(this_dir, 'summaries')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    learning_rate = 0.25
    with summary_writer.as_default():
        for iteration in range(50):  # Train for 50 iterations.
            # Full-batch gradient descent.
            with tf.GradientTape() as tape:
                # The variables in out model are watched automatically.
                # Compute the outputs of the model.
                estimates = model.forward_pass(input_data)
                # Measure the error
                error = mean_absolute_error(target_data, estimates)

            # Compute the derivative of the error w.r.t. the parameters.
            dweights, dbias = tape.gradient(error, [model.weights, model.bias])

            # Report some values.
            tf.summary.scalar('training error', error, step=iteration)
            tf.summary.histogram('weights', model.weights, step=iteration)
            summary_writer.flush()  # Write every iteration.

            # Vanilla gradient descent update rule.
            model.weights.assign(model.weights - learning_rate * dweights)
            model.bias.assign(model.bias - learning_rate * dbias)


if __name__ == '__main__':
    # Remove the summaries dir before every run.
    summary_dir = os.path.join(os.path.dirname(__file__), 'summaries')
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

    main()
