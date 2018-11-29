"""Minimal example that shows how summaries are used.

Observe the result in the web browser (localhost:6006) after starting the
tensorboard:

$ tensorboard --logdir=./experiments/
"""
import numpy as np
import tensorflow as tf


np.random.seed(12123)


# Create a placeholder for our data.
input_data = tf.placeholder(tf.float32, [1000, 10])

# Create some summaries.
tf.summary.scalar('Mean of data.', tf.reduce_mean(input_data))
tf.summary.histogram('Data', input_data)

# Merge all summaries. <- tensorflow magic op.
all_summaries_op = tf.summary.merge_all()

# Create a writer for storing the summaries on disk for tensorboard to find.
summary_writer = tf.summary.FileWriter('./experiments/tf_summary_example')

# Let's create some summary events.
with tf.Session() as session:
    for step in range(100):
        # Generate some random data.
        data = np.random.uniform(0, 10, size=[1000, 10])

        # Compute the summary values.
        all_summaries = session.run(
            all_summaries_op, feed_dict={input_data: data})

        # Write the summaries, dont forget the step ('x-coordinate' in plot).
        summary_writer.add_summary(all_summaries, step)
