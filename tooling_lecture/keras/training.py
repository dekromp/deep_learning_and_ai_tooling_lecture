"""Shows the usage of the functional API of keras for training."""
import numpy as np
import tensorflow as tf


# Set the random seeds to make the outputs deterministic.
np.random.seed(1)
tf.random.set_seed(1)


def main():
    """Train a model on data."""
    x, y = load_data()
    model = build_model()

    # Configure the model for training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
        loss=tf.keras.losses.binary_crossentropy,
    )

    # Train the model.
    model.fit(x=x, y=y, batch_size=8, epochs=10, verbose=2)
    """
    Train on 100 samples
    Epoch 1/10
    100/100 - 1s - loss: 0.7048 - binary_accuracy: 0.4900
    Epoch 2/10
    100/100 - 0s - loss: 0.6915 - binary_accuracy: 0.5200
    Epoch 3/10
    100/100 - 0s - loss: 0.6779 - binary_accuracy: 0.5800
    Epoch 4/10
    100/100 - 0s - loss: 0.6664 - binary_accuracy: 0.6400
    Epoch 5/10
    100/100 - 0s - loss: 0.6544 - binary_accuracy: 0.6800
    Epoch 6/10
    100/100 - 0s - loss: 0.6434 - binary_accuracy: 0.7100
    Epoch 7/10
    100/100 - 0s - loss: 0.6333 - binary_accuracy: 0.7200
    Epoch 8/10
    100/100 - 0s - loss: 0.6228 - binary_accuracy: 0.7400
    Epoch 9/10
    100/100 - 0s - loss: 0.6131 - binary_accuracy: 0.7500
    Epoch 10/10
    100/100 - 0s - loss: 0.6032 - binary_accuracy: 0.7600
    """


def load_data():
    """Generate some random binary classification data that is easy to fit.

    Returns
    -------
    numpy.ndarray
        The feature data.
    numpy.ndarray
        The target data.

    """
    input_data = np.random.randn(100, 3)
    target_data = input_data.dot(np.random.uniform(1, 2, size=(3, 1))) + 2.1
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
        shape=(3,)
    )  # shape does not include the batch size.
    layer1 = tf.keras.layers.Dense(5, activation=tf.keras.activations.tanh)
    layer2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    h = layer1(input_x)
    output = layer2(h)
    return tf.keras.Model(inputs=[input_x], outputs=[output])


if __name__ == '__main__':
    main()
