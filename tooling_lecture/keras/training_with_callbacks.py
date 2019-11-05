"""Shows the usage of the functional API of keras for training."""
import os
import shutil

import numpy as np
import tensorflow as tf


# Set the random seeds to make the outputs deterministic.
np.random.seed(1)
tf.random.set_seed(1)


def main(jobdir):
    """Train a model on data."""
    x, y = load_data()
    model = build_model()

    # Configure the model for training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
        loss=tf.keras.losses.binary_crossentropy,
    )

    # Configure the callbacks for checkpointing the model and monitoring the
    # training loss and metrics in tensorboard.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(jobdir, 'model.h5'), monitor='loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=jobdir, histogram_freq=1, write_graph=True
        ),
    ]

    # Train the model.
    model.fit(
        x=x, y=y, batch_size=8, epochs=100, verbose=2, callbacks=callbacks
    )


def load_data():
    """Generate some random binary classification data that is easy to fit.

    Returns
    -------
    numpy.ndarray
        The feature data.
    numpy.ndarray
        The target data.

    """
    input_data = np.random.randn(1000, 30)
    target_data = input_data.dot(np.random.uniform(1, 2, size=(30, 1))) + 2.1
    target_data -= np.mean(target_data)
    target_data[target_data > 0] = 1
    target_data[target_data <= 0] = 0
    return input_data, target_data


def build_model():
    """Build the model.

    Returns
    -------
    tensorflow.keras.Model
        The model.

    """
    input_x = tf.keras.Input(
        shape=(30,)
    )  # shape does not include the batch size.
    layer1 = tf.keras.layers.Dense(5, activation=tf.keras.activations.tanh)
    layer2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    h = layer1(input_x)
    output = layer2(h)
    return tf.keras.Model(inputs=[input_x], outputs=[output])


if __name__ == '__main__':
    jobdir = os.path.join(
        os.path.dirname(__file__), 'training_with_callbacks_results'
    )
    # Clean up before starting the training.
    if os.path.exists(jobdir):
        shutil.rmtree(jobdir)
    main(jobdir)
