"""Ray can be used to implement grid search.

You need to add the tooling_lecture project to your PYTHONPATH for this code
to work. If you use conda, you can use conda develop <path>/tooling_lecture
to make the project visible for python.

"""
import os

import ray
from ray import tune

from tooling_lecture.hypertune import keras_example


def trainable(config):
    """A wrapper for tune.

    Parameters
    ----------
    config : dict
        The configuration of the run. Contains explicit hyperparmeters.

    """
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    l2 = config['l2']
    learning_rate = config['learning_rate']
    experiment_dir = os.path.join(
        os.path.abspath(__file__),
        'experiments',
        'ray_modelV1_lr[%s]_l2[%s]' % (learning_rate, l2),
    )
    keras_example.main(
        num_epochs=num_epochs,
        batch_size=batch_size,
        l2_factor=l2,
        learning_rate=learning_rate,
        experiment_dir=experiment_dir,
        verbose=2,
    )


if __name__ == '__main__':
    ray.init()
    num_epochs = 200
    batch_size = 8

    tune_config = dict(
        l2=tune.grid_search([1e-5, 1e-4, 1e-3]),
        learning_rate=tune.grid_search([1e-2, 1e-1, 1]),
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    results = tune.run(
        trainable, name='simple_hp', config=tune_config, num_samples=1
    )
    print(results.dataframe())
