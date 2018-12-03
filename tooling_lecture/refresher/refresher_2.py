"""Simple example script that shows basic operations of tensorflow."""
import numpy as np
import tensorflow as tf


# Fix the random seeds to make the computations reproducable.
tf.set_random_seed(12345)
np.random.seed(12321)

# Create a placeholder for feeding inputs in the graph.
input_x = tf.placeholder(tf.float32, [None, 3], name='features')

# Create a variable.
w = tf.get_variable(
    'weights', [3, 1], initializer=tf.glorot_uniform_initializer())

# Perform some computation steps.
output = tf.matmul(input_x, w)
output = tf.reshape(output, [-1])  # Flatten the outputs.

# Create a target placeholder and define the loss computation.
input_y = tf.placeholder(tf.float32, [None], name='target')
# Mean squared error.
loss = tf.reduce_mean(tf.square(output - input_y))

# Define the update operation (stochastic gradient descent).
update_op = tf.assign(w, w - 0.01 * tf.gradients(loss, w)[0])

# Generate some random training data.
x = np.random.randn(100, 3)
unknown_w = np.array([0.3, -0.21, 0.8])
y = np.dot(x, unknown_w)

# Execute the graph on some random data.
batch_size = 8
num_epochs = 15
with tf.Session() as session:
    # Boilerplate code that initializes all variables in the graph (just w).
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

        print('Epoch %d; TrainLoss: %.4f' % (epoch + 1, np.mean(epoch_losses)))

    print('Found parameters: %s' % str(w.eval().reshape(-1)))
    print('True parameters: %s' % str(unknown_w))
