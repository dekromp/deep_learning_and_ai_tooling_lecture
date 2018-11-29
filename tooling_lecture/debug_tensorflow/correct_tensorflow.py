"""A simple feedforward network that is trained on data."""
import numpy as np
import tensorflow as tf


# Set all random seeds for reproducebility.
np.random.seed(1212)
tf.set_random_seed(112321)


def main():
    """Main function to execute the experiment."""
    # Create some random data.
    x1, x2, y = load_data()

    # Build forward pass through the network.
    input_x1, input_x2, output = build_forward_pass()
    # Build loss and the update operations.
    update_op, loss, input_y = build_objective(output)
    # Train the model on the data.
    inputs = (input_x1, input_x2, input_y)
    data = (x1, x2, y)
    train_model(inputs, data, loss, update_op)


def build_forward_pass():
    """Build the forward pass of the model.

    Returns
    -------
    input_x1 : :class:`tf.tensor`
        The input for the first feature set.
    input_x2 : :class:`tf.tensor`
        The input for the second feature set.
    output : :class:`tf.tensor`
        The output of the model.

    """
    with tf.name_scope('forward_pass'):
        input_x1 = tf.placeholder(tf.float32, [None, 10], name='input_x1')
        input_x2 = tf.placeholder(tf.float32, [None, 20], name='input_x2')
        h1 = dense_layer(input_x1, 'layer1', 5, activation=relu)
        h2 = dense_layer(input_x2, 'layer2', 7, activation=relu)
        h = tf.concat([h1, h2], axis=-1)
        logits = dense_layer(h, 'output_layer', 1)
        output = sigmoid(logits)

    return input_x1, input_x2, output


def build_objective(output):
    """Build the graph for the objective and parameter update.

    Parameters
    ----------
    output : :class:`tf.tensor`
        The tensor that represents the output of the model.

    Returns
    -------
    update_op : :class:`tf.tensor`
        The tensor that represents the output of the update operation.
    loss : :class:`tf.tensor`
        The tensor that represents the outputof the loss.
    input_y : :class:`tf.tensor`
        The input tensor for the targets.

    """
    # Build the loss.
    with tf.name_scope('loss'):
        # Flatten the output.
        output = tf.reshape(output, [-1])
        # Create an input for the inputs
        input_y = tf.placeholder(tf.float32, [None], name='input_y')

        # Compute the loss (binary cross entropy)
        epsilon = 1e-7  # for numerical stability.
        loss = -(tf.multiply(input_y, tf.log(output + epsilon)) +
                 tf.multiply(1 - input_y, tf.log(1 - output + epsilon)))
        loss = tf.reduce_mean(loss, name='loss_out')

    # Build the update op.
    with tf.name_scope('update_op'):
        learning_rate = 0.01
        grads = tf.gradients(loss, tf.trainable_variables())
        update_ops = []
        for grad, v in zip(grads, tf.trainable_variables()):
            update_ops.append(tf.assign(v, v - learning_rate * grad))
        update_op = tf.group(*update_ops)

    return update_op, loss, input_y


def relu(x):
    """Relu activation function.

    Parameters
    ----------
    x : :class:`tf.tensor`
        The input to this op.

    Returns
    -------
    activated : :class:`tf.tensor`
        The activated input.

    """
    return tf.maximum(x, 0)


def sigmoid(x):
    """Sigmoid activation function.

    Parameters
    ----------
    x : :class:`tf.tensor`
        The input to this op.

    Returns
    -------
    activated : :class:`tf.tensor`
        The activated input.

    """
    # Make sure that the values of x are not too small/big.
    x = tf.clip_by_value(x, -80, 80)

    negative = tf.less(x, 0.0)
    activation = tf.where(
        negative, tf.exp(x) / (1.0 + tf.exp(x)), 1.0 / (1.0 + tf.exp(-x)))
    return activation


def dense_layer(x, layer_name, units, activation=None):
    """Apply a dense layer to an input.

    Parameters
    ----------
    x : :class:`tf.tensor`
        The input to this op.
    layer_name : name
        The name scope of the variables.
    units : int
        The number of hidden units.
    activation : callable or `None`, optional
        The activation function applied on the outputs.
        No activation function is used when `None`.
        Defaults to `None`.

    Returns
    -------
    h : :class:`tf.tensor`
        The output of this layer.

    """
    with tf.variable_scope(layer_name):
        # Initialize the parameters of the layer.
        weights = tf.get_variable(
            'W', dtype=tf.float32,
            shape=[x.get_shape()[1], units], trainable=True,
            initializer=tf.initializers.truncated_normal(
                stddev=0.01, mean=0.0))
        b = tf.get_variable(
            'b', dtype=tf.float32, shape=[units], trainable=True,
            initializer=tf.constant_initializer(0.0))

        # Compute the outputs.
        output = tf.matmul(x, weights) + b
        # Apply activation function if desired.
        if activation is not None:
            output = activation(output)
        return output


def train_model(inputs, data, loss, update_op):
    """Train the model on some input data.

    Parameters
    ----------
    inputs : tuple
        The input tensors for training.
    data : tuple
        The data the model should be trained on. Must have the same order as
        inputs.
    loss : :class:`tf.tensor`
        The tensor that represents the output of the loss. Used for computing
        the training loss.
    update_op : :class:`tf.tensor`
        The tensor that represents the output of the update operation.

    """
    input_x1, input_x2, input_y = inputs
    x1, x2, y = data

    # Execute the graph on some random data.
    with tf.Session() as session:
        # Initialize all variables in the graph.
        session.run(tf.global_variables_initializer())
        num_epochs = 100
        batch_size = 8
        for epoch in range(num_epochs):  # Train for 15 epochs.
            # Shuffle the training data.
            shuffle_idx = np.random.permutation(np.arange(len(x1)))
            x1 = x1[shuffle_idx]
            x2 = x2[shuffle_idx]
            y = y[shuffle_idx]

            # Train the model on batches of data with SGD.
            epoch_losses = []
            for i in range(0, len(x1), batch_size):
                batch_loss, _ = session.run(
                    [loss, update_op],
                    feed_dict={input_x1: x1[i: i + batch_size],
                               input_x2: x2[i: i + batch_size],
                               input_y: y[i: i + batch_size]})
                epoch_losses += [batch_loss]

            print('Epoch %d; TrainLoss: %.4f'
                  % (epoch + 1, np.mean(epoch_losses)))


def load_data():
    """Generate some random data for training.

    Returns
    -------
    x1 : :class:`numpy.ndarray`
        The dummy data for the first feature set.
    x2 : :class:`numpy.ndarray`
        The dummy data for the second feature set.
    y : :class:`numpy.ndarray`
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
