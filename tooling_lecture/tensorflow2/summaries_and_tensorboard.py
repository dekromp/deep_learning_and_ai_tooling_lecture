"""Show the usage of tf.summary for visualizing things in Tensorboard."""
import os
import shutil

import numpy as np
import tensorflow as tf


def scalar():
    """Create some summary files Tensorboard understands."""
    # Some data we want to visualize.
    data = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5])

    # Get the directory this file lives in.
    this_dir = os.path.dirname(__file__)

    # Define a writer that writes files that Tensorboard understand.
    summary_dir = os.path.join(this_dir, 'summaries')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    with summary_writer.as_default():
        for i in range(len(data)):
            tf.summary.scalar('my_scalar', data[i], step=i)
            summary_writer.flush()


def histogram():
    """Create some summary files Tensorboard understands."""
    # Some data we want to visualize.
    data = [np.random.normal(0, 1 - i * 0.15, size=1000) for i in range(5)]
    # Get the directory this file lives in.
    this_dir = os.path.dirname(__file__)

    # Define a writer that writes files that Tensorboard understand.
    summary_dir = os.path.join(this_dir, 'summaries')
    summary_writer = tf.summary.create_file_writer(summary_dir)

    with summary_writer.as_default():
        for i in range(len(data)):
            tf.summary.histogram('my_histogram', data[i], step=i)
            summary_writer.flush()


if __name__ == '__main__':
    # Remove the summaries dir before every run.
    summary_dir = os.path.join(os.path.dirname(__file__), 'summaries')
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

    scalar()
    histogram()
