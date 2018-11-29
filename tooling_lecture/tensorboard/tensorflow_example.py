"""A simple feedforward network that is trained on data."""
from __future__ import print_function
from builtins import range

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug


# Set all random seeds for reproducebility.
np.random.seed(1212)
tf.set_random_seed(112321)


def main(num_epochs, batch_size, learning_rate, experiment_dir, debug):
    """Train a simple model on random data.

    Parameters
    ----------
    num_epochs : int
        The number of epochs the model is trained.
    batch_size : int
        The batch size used for SGD.
    learning_rate : float
        The learning rate used for SGD.
    experiment_dir : str
        The path to the experiment directory where the summaries will be saved.
    debug : bool
        Whether or not the script is debugged with the tensorflow debugger.

    """
    # Create some random data.
    train_dataset, val_dataset = split_dataset(load_data(), 0.9)

    # Build forward pass through the network.
    input_x1, input_x2, output = build_forward_pass()
    # Build loss and the update operations.
    update_op, loss, input_y = build_objective(output, learning_rate)

    # Train the model on the data.
    inputs = (input_x1, input_x2, input_y)
    train_model(inputs, train_dataset, val_dataset, loss, update_op,
                num_epochs, batch_size, experiment_dir, debug)


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

    # Compute a historgram over the outputs.
    tf.summary.histogram('model_outputs', output)

    return input_x1, input_x2, output


def build_objective(output, learning_rate):
    """Build the graph for the objective and parameter update.

    Parameters
    ----------
    output : :class:`tf.tensor`
        The tensor that represents the output of the model.
    learning_rate : float
        The learning rate used for SGD.

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
        """
        Really nasty bug here: tensor multiply does broadcasting which causes
        input_y * output to be of shape batch x batch.
        """
        # Flatten the output.
        output = tf.reshape(output, [-1])
        # Create an input for the inputs
        input_y = tf.placeholder(tf.float32, [None], name='input_y')

        # Compute the loss (binary cross entropy)
        epsilon = 1e-7  # for numerical stability.
        loss = -(tf.multiply(input_y, tf.log(output + epsilon)) +
                 tf.multiply(1 - input_y, tf.log(1 - output + epsilon)))
        loss = tf.reduce_mean(loss, name='loss_out')

    # Monitor the loss.
    tf.summary.scalar('epoch_val_loss', loss)

    # Build the update op.
    with tf.name_scope('update_op'):
        grads = tf.gradients(loss, tf.trainable_variables())
        update_ops = []
        for grad, v in zip(grads, tf.trainable_variables()):
            update_ops.append(tf.assign(v, v - learning_rate * grad))
            # Add a summary for the gradients.
            tf.summary.histogram('grad_%s' % v.name, v)
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
        # Compute a historgram over the weights.
        tf.summary.histogram('%s weights' % layer_name, weights)
        tf.summary.image(
            '%s weights' % layer_name,
            tf.reshape(weights, [1, x.get_shape()[1], units, 1]))
        b = tf.get_variable(
            'b', dtype=tf.float32, shape=[units], trainable=True,
            initializer=tf.constant_initializer(0.0))

        # Compute the outputs.
        output = tf.matmul(x, weights) + b
        # Apply activation function if desired.
        if activation is not None:
            output = activation(output)
        return output


def train_model(inputs, train_data, val_data, loss, update_op, num_epochs,
                batch_size, experiment_dir, debug):
    """Train the model on some input data.

    Parameters
    ----------
    inputs : tuple
        The input tensors for training.
    train_data : tuple
        The data the model should be trained on. Must have the same order as
        inputs.
    val_data : tuple
        The data used for monitoring.
    loss : :class:`tf.tensor`
        The tensor that represents the output of the loss. Used for computing
        the training loss.
    update_op : :class:`tf.tensor`
        The tensor that represents the output of the update operation.
    num_epochs : int
        The number of epochs the model is trained.
    batch_size : int
        The batch size used for SGD.
    experiment_dir : str
        The path to the experiment directory where the summaries will be saved.
    debug : bool
        Whether or not the script is debugged with the tensorflow debugger.

    """
    # Quick hack to add metrics computed in python.
    train_loss_plh = tf.placeholder(tf.float32, [], name='train_loss')
    tf.summary.scalar('epoch_loss', train_loss_plh)

    # Merge all summaries and initialize the summary writer.
    all_summaries_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(experiment_dir)

    input_x1, input_x2, input_y = inputs
    x1, x2, y = train_data
    x1_val, x2_val, y_val = val_data

    # Execute the graph on some random data.
    with tf.Session() as session:
        # Trigger debugging if desired.
        if debug:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            tf.logging.set_verbosity(tf.logging.ERROR)

        # Add the graph to the summaries.
        summary_writer.add_graph(session.graph)
        # Initialize all variables in the graph.
        session.run(tf.global_variables_initializer())
        epoch_train_loss = np.inf
        for epoch in range(num_epochs):  # Train for 15 epochs.
            # Compute the summaries on the validation data.
            summaries = session.run(
                all_summaries_op,
                feed_dict={input_x1: x1_val, input_x2: x2_val, input_y: y_val,
                           train_loss_plh: epoch_train_loss})
            # Write the summaries for this epoch.
            summary_writer.add_summary(summaries, epoch)

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

            epoch_train_loss = np.mean(epoch_losses)
            print('Epoch %d; TrainLoss: %.4f'
                  % (epoch + 1, epoch_train_loss))


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


def split_dataset(dataset, size):
    """Split the arrays of a dataset in two parts.

    Parameters
    ----------
    dataset : tuple of :class:`numpy.ndarray`
        The arrays of the dataset.
    size : float
        [0, 1], the relative size of the first half of the split.

    Returns
    -------
    dataset1 : tuple of :class:`numpy.ndarray`
        The first part of the split.
    dataset2 : tuple of :class:`numpy.ndarray`
        The second part of the split.

    """
    split_idx = int(len(dataset[0]) * size)
    dataset1 = []
    dataset2 = []
    for data in dataset:
        dataset1 += [data[:split_idx]]
        dataset2 += [data[split_idx:]]

    return tuple(dataset1), tuple(dataset2)


if __name__ == '__main__':
    # Simple commandline interface for configuring the execution.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')
    parser.add_argument(
        '-ep', '--num_epochs', type=int, default=500,
        help='Number of epochs the model should be trained.')
    parser.add_argument(
        '-bs', '--batch-size', type=int, default=8,
        help='The batch size used in every training iteration.')
    parser.add_argument(
        '-lr', '--learning-rate', type=float, default=0.01,
        help='The learning rate for SGD.')
    parser.add_argument(
        '-d', '--experiment-dir', type=str, default='./experiments/default',
        help='The path to the experiment directory.')
    parser.add_argument(
        '--debug',
        help='Debug the script with the tensorflow debugger.',
        action='store_true')
    args = parser.parse_args()
    config = vars(args)
    main(**config)
