"""Simple example script that shows basic operations of tensorflow.

Same as refresher_2 but the code has been structured, documented and contains
a command line interface to change the run configuration.

"""
import argparse
import numpy as np
import tensorflow as tf


# Fix the random seeds to make the computations reproducable.
tf.set_random_seed(12345)
np.random.seed(12321)

# Constants of the experiments.
unknown_true_w = np.array([0.3, -0.21, 0.8])


def main(num_epochs, batch_size, learning_rate):
    """Train a simple model on random data.

    Parameters
    ----------
    num_epochs : int
        The number of epochs the model is trained.
    batch_size : int
        The batch size used for SGD.
    learning_rate : float
        The learning rate used for SGD.

    """
    # Generate some random training data.
    x = np.random.randn(100, 3)
    y = np.dot(x, unknown_true_w)

    # Build forward pass.
    input_x, output = build_forward_pass()
    # Build the update op with respect to the objective.
    update_op, loss, input_y = build_objective(output, learning_rate)
    # Fit the model on the input data.
    inputs = (input_x, input_y)
    data = (x, y)
    train_model(inputs, data, loss, update_op, batch_size, num_epochs)


def build_forward_pass():
    """Build the forward pass of the model.

    Returns
    -------
    input_x : tf.tensor
        The input tensor for the features.
    output: tf.tensor
        The output of the forward pass.

    """
    # Create a placeholder for feeding inputs in the graph.
    input_x = tf.placeholder(tf.float32, [None, 3], name='features')

    # Create a variable, which will be automatically added to the trainable
    # variables collection.
    w = tf.get_variable(
        'weights', [3, 1], initializer=tf.glorot_uniform_initializer())

    # Perform some computation steps.
    output = tf.matmul(input_x, w)
    output = tf.reshape(output, [-1])  # Flatten the outputs.

    return input_x, output


def build_objective(output, learning_rate):
    """Build the graph for the objective and parameter update.

    Parameters
    ----------
    output : tf.tensor
        The tensor that represents the output of the model.
    learning_rate : float
        The learning rate used for SGD.

    Returns
    -------
    update_op : tf.tensor
        The tensor that represents the output of the update operation.
    loss : tf.tensor
        The tensor that represents the outputof the loss.
    input_y : tf.tensor
        The input tensor for the targets.

    """
    # Create a target placeholder and define the loss computation.
    input_y = tf.placeholder(tf.float32, [None], name='target')
    # Mean squared error.
    loss = tf.reduce_mean(tf.square(output - input_y))

    # Define the update operation (stochastic gradient descent).
    w = tf.trainable_variables()[0]
    update_op = tf.assign(w, w - 0.01 * tf.gradients(loss, w)[0])

    return update_op, loss, input_y


def train_model(inputs, data, loss, update_op, batch_size, num_epochs):
    """Train the model on some input data.

    Parameters
    ----------
    inputs : tuple
        The input tensors for training.
    data : tuple
        The data the model should be trained on. Must have the same order as
        inputs.
    loss : tf.tensor
        The tensor that represents the output of the loss. Used for computing
        the training loss.
    update_op : tf.tensor
        The tensor that represents the output of the update operation.
    batch_size : int
        The batch size used for SGD.
    num_epochs : int
        The number of epochs the model is trained.

    """
    input_x, input_y = inputs
    x, y = data

    # Execute the graph on some random data.
    with tf.Session() as session:
        # Boilerplate code that initializes all variables in the graph (w).
        session.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):  # Train for 15 epochs.
            # Shuffle the training data.
            shuffle_idx = np.random.permutation(np.arange(len(x)))
            x = x[shuffle_idx]
            y = y[shuffle_idx]

            # Train the model on batches of data with SGD.
            epoch_losses = []
            for i in range(0, len(x), batch_size):
                batch_loss, _ = session.run(
                    [loss, update_op],
                    feed_dict={input_x: x[i: i + batch_size],
                               input_y: y[i: i + batch_size]})
                epoch_losses += [batch_loss]

            print('Epoch %d; TrainLoss: %.6f'
                  % (epoch + 1, np.mean(epoch_losses)))

        w = tf.trainable_variables()[0]
        print('Found parameters: %s' % str(w.eval().reshape(-1)))
        print('True parameters: %s' % str(unknown_true_w))


if __name__ == '__main__':
    # Simple commandline interface for configuring the execution.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')
    parser.add_argument(
        '-ep', '--num_epochs', type=int, default=15,
        help='Number of epochs the model should be trained.')
    parser.add_argument(
        '-bs', '--batch-size', type=int, default=8,
        help='The batch size used in every training iteration.')
    parser.add_argument(
        '-lr', '--learning-rate', type=float, default=0.01,
        help='The learning rate for SGD.')
    args = parser.parse_args()
    config = vars(args)
    main(**config)
