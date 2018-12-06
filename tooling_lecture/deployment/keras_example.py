"""A simple feedforward network that is trained on data."""
from __future__ import print_function

import os

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Set all random seeds for reproducebility.
np.random.seed(1212)
tf.set_random_seed(112321)


def main(num_epochs, batch_size, learning_rate, experiment_dir):
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

    """
    # Create some random data.
    dataset = load_data()
    x1, x1_val, x2, x2_val, y, y_val = train_test_split(
        *dataset, test_size=0.1)

    # Build forward pass through the network.
    input_x1, input_x2, output = build_forward_pass()

    # Build the model with keras.
    model = tf.keras.Model([input_x1, input_x2], [output])
    # Build loss and the update operations.
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(optimizer, loss=tf.keras.losses.binary_crossentropy)

    # Define some callbacks.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        experiment_dir, write_graph=True, write_images=True,
        histogram_freq=1)
    # For saving the model.
    checkpointing_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(experiment_dir, 'model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min')
    callbacks = [tensorboard_callback, checkpointing_callback]

    # Train the model on the data.
    model.fit(x=[x1, x2], y=y, batch_size=batch_size, epochs=num_epochs,
              validation_data=([x1_val, x2_val], y_val), verbose=2,
              callbacks=callbacks)


def build_forward_pass():
    """Build the forward pass of the model.

    Returns
    -------
    input_x1 : tf.tensor
        The input for the first feature set.
    input_x2 : tf.tensor
        The input for the second feature set.
    output : tf.tensor
        The output of the model.

    """
    with tf.name_scope('forward_pass'):
        input_x1 = tf.keras.Input([10], name='input_x1')
        input_x2 = tf.keras.Input([20], name='input_x2')
        h1 = tf.keras.layers.Dense(
            units=5, activation=tf.nn.relu, name='layer1')(input_x1)
        h2 = tf.keras.layers.Dense(
            units=7, activation=tf.nn.relu, name='layer2')(input_x2)
        h = tf.keras.layers.Concatenate(axis=-1)([h1, h2])
        output = tf.keras.layers.Dense(
            units=1, activation=tf.nn.sigmoid, name='output_layer')(h)

    return input_x1, input_x2, output


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
        '-d', '--experiment-dir', type=str,
        default='./experiments/keras_example',
        help='The path to the experiment directory.')
    args = parser.parse_args()
    config = vars(args)
    main(**config)
