"""Ray can be used to implement grid search.

You need to add the tooling_lecture project to your PYTHONPATH for this code
to work. If you use conda, you can use conda develop <path>/tooling_lecture
to make the project visible for python.

"""
import os

import numpy as np
import ray
from ray import tune

from tooling_lecture.hypertune import keras_example


def pbt_scheduler():
    """Initialize a PBT scheduler.

    In case of pbt, num_samples determines the initial population.

    """
    pbt = tune.schedulers.PopulationBasedTraining(
        time_attr='training_iteration',
        metric='mean_accuracy',
        mode='max',
        perturbation_interval=5,
        hyperparam_mutations=dict(
            l2=lambda: np.exp(
                np.random.uniform(np.log(0.01), np.log(0.00001))
            ),
            learning_rate=[1e-2, 1e-1, 1],
        ),
    )
    return pbt


class MyTrainable(tune.Trainable):
    """Ray Trainable for traning and tuning the model."""

    def _setup(self, config):
        """Setup the model."""
        import tensorflow as tf  # There is an issue with ray + tf.

        learning_rate = config['learning_rate']
        self._model = keras_example.build_model(config['l2'])

        # Build loss and the update operations.
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
        self._model.compile(
            optimizer,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['binary_accuracy'],
        )
        self._batch_size = config['batch_size']

    def _train(self):
        """Perform a training iteration."""
        dataset = keras_example.load_data()
        x1, x1_val, x2, x2_val, y, y_val = keras_example.train_test_split(
            *dataset, test_size=0.1
        )
        hist = self._model.fit(
            x=[x1, x2],
            y=y,
            epochs=1,
            batch_size=batch_size,
            validation_data=([x1_val, x2_val], y_val),
            verbose=2,
        )
        hist = hist.history
        val_loss = hist['val_loss'][0]
        val_accuracy = hist['val_binary_accuracy'][0]
        return dict(mean_accuracy=val_accuracy, mean_loss=val_loss)

    def _save(self, tmp_checkpoint_dir):
        """Checkpoint the model."""
        savefile = os.path.join(tmp_checkpoint_dir, 'model.h5')
        self._model.save(savefile, include_optimizer=True, save_format='h5')
        return savefile

    def _restore(self, checkpoint):
        """Restore the model from disk."""
        import tensorflow as tf  # There is an issue with ray + tf.

        self._model = tf.keras.models.load_model(checkpoint)


if __name__ == '__main__':
    ray.init()
    num_epochs = 200
    batch_size = 8

    tune_config = dict(
        l2=tune.grid_search([1e-5, 1e-4, 1e-3]),
        learning_rate=tune.grid_search([1e-2, 1e-1, 1]),
        batch_size=batch_size,
    )
    results = tune.run(
        MyTrainable,
        name='adv_hp',
        config=tune_config,
        stop=dict(training_iteration=num_epochs),
        num_samples=1,
        scheduler=pbt_scheduler(),
    )
    print(results.dataframe())
