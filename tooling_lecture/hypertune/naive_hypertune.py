"""Simple way to call python scripts."""
import os
import subprocess


num_epochs = 200
batch_size = 8
path = './experiments'
try:
    for l2_factor in [1e-4, 1e-3, 1e-2]:
        for learning_rate in [1e-2, 1e-1, 1]:
            subprocess.call([
                'python', 'keras_example.py',
                '-ep', str(num_epochs),
                '-bs', str(batch_size),
                '-lr', str(learning_rate),
                '-l2', str(l2_factor),
                '-d', os.path.join(
                    path, 'modelV1_lr[%s]_l2[%s]'
                    % (learning_rate, l2_factor))])
except subprocess.CalledProcessError as e:
    print(e)
