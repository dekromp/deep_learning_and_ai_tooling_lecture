"""A simple feedforward network that is trained on data."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


tf.enable_eager_execution()


# Set all random seeds for reproducebility.
np.random.seed(1212)
tf.set_random_seed(112321)


def main():
    """Main function to execute the experiment."""
    # Create some random data.
    x1, x2, y = load_data()

    # All variables need to be created in the context of a EagerVariableStore
    # otherwise tensorflow collections will not work.
    with tfe.EagerVariableStore().as_default():
        # Build forward pass through the network.
        dense_layer1 = DenseLayer('layer1', 5, x1.shape[1], activation=relu)
        dense_layer2 = DenseLayer('layer2', 7, x2.shape[1], activation=relu)
        dense_layer3 = DenseLayer('layer3', 1, 12, activation=sigmoid)
    all_params = tf.trainable_variables()  # Some tensorflow magic.

    # Train the model for a couple of epochs.
    num_epochs = 100
    batch_size = 8
    learning_rate = 0.01
    for n in range(num_epochs):
        # Shuffle the training data.
        sidx = np.random.permutation(np.arange(x1.shape[0]))
        x1 = x1[sidx]
        x2 = x2[sidx]
        y = y[sidx]

        # Train the model on batches of data with SGD.
        epoch_losses = []
        for i in range(0, len(x1), batch_size):
            # Build the batches.
            batch_x1 = x1[i: i + batch_size]
            batch_x2 = x2[i: i + batch_size]
            batch_y = y[i: i + batch_size]

            # The gradient tape is specific for eager execution. It keeps track
            # of all the computed outputs in the graph which will be used later
            # to compute the gradients. Note that some magic is happening.
            # Every variable initialized with `trainable=True` (default) is
            # automatically watched but other tensors can be watched, too.
            # See https://www.tensorflow.org/api_docs/python/tf/GradientTape.
            with tf.GradientTape() as tape:
                # Compute the forward pass using the batches.
                h1 = dense_layer1(batch_x1)
                h2 = dense_layer2(batch_x2)
                h = tf.concat([h1, h2], axis=-1)
                output = tf.reshape(dense_layer3(h), [-1])
                # Compute the binary cross entropy loss.
                loss = -(tf.multiply(batch_y, tf.log(output)) +
                         tf.multiply(1 - batch_y, tf.log(1 - output)))
                loss = tf.reduce_mean(loss)

            # Compute the gradients and update the variables.
            grads = tape.gradient(loss, all_params)
            for grad, v in zip(grads, all_params):
                tf.assign(v, v - learning_rate * grad)
            epoch_losses += [loss]
        print('Epoch: %d; Training Loss: %.4f.'
              % (n + 1, np.mean(epoch_losses)))


class DenseLayer(object):
    """Own implementation of a dense layer.

    Parameters
    ----------
    layer_name : str
        The name of the layer, used as scope name.
    units : int
        Number of hidden units.
    input_size : int
        The size of the input.
    activation : callable or `None`, optional
        A function that computes an activation.
        If `None` no activation is used.
        Defaults to `None`.
    """

    def __init__(self, layer_name, units, input_size, activation=None):  # noqa
        self.layer_name = layer_name
        self.activation = activation
        with tf.variable_scope(layer_name):
            self.weights = tf.get_variable(
                'W', dtype=tf.float32,
                shape=[input_size, units], trainable=True,
                initializer=tf.initializers.truncated_normal(
                    stddev=0.01, mean=0.0))
            self.b = tf.get_variable(
                'b', dtype=tf.float32, shape=[units], trainable=True,
                initializer=tf.constant_initializer(0.0))

    def __call__(self, x):
        """Compute the output of the layer.

        Parameters
        ----------
        x : tf.tensor or numpy.ndarray
            The input to this layer.

        Returns
        -------
        h : tf.tensor
            The output of this layer.

        """
        h = tf.matmul(x, self.weights) + self.b
        if self.activation is not None:
            h = self.activation(h)
        return h


def relu(x):
    """Relu activation function.

    Parameters
    ----------
    x : tf.tensor
        The input to this op.

    Returns
    -------
    activated : tf.tensor
        The activated input.

    """
    return tf.maximum(x, 0)


def sigmoid(x):
    """Sigmoid activation function.

    Parameters
    ----------
    x : tf.tensor
        The input to this op.

    Returns
    -------
    activated : tf.tensor
        The activated input.

    """
    # Make sure that the values of x are not too small/big.
    x = tf.clip_by_value(x, -80, 80)

    negative = tf.less(x, 0.0)
    activation = tf.where(
        negative, tf.exp(x) / (1.0 + tf.exp(x)), 1.0 / (1.0 + tf.exp(-x)))
    return activation


def load_data():
    """Generate some random data for training.

    Returns
    -------
    x1 : numpy.ndarray
        The dummy data for the first feature set.
    x2 : numpy.ndarray
        The dummy data for the second feature set.
    y : numpy.ndarray
        The data for the targets.

    """
    x1 = np.random.randn(1000, 10).astype(np.float32)
    x2 = np.random.randn(1000, 20).astype(np.float32)
    x = np.concatenate([x1, x2], axis=-1)
    w = np.random.randn(30, 1).astype(np.float32)
    y = x.dot(w).reshape(-1)
    y[y > 0] = 1.0
    y[y <= 0] = 0.0
    return x1, x2, y


if __name__ == '__main__':
    main()
