"""Ray can be used to parallelize our grid search implementation."""
import ray

from tooling_lecture.hypertune import keras_example


ray.init()


@ray.remote
def keras_example_main(num_epochs, batch_size, l2_factor, learning_rate,
                       experiment_dir):
    """A wrapper for the main function of the keras example."""
    keras_example.main(
        num_epochs, batch_size, l2_factor, learning_rate, experiment_dir,
        debug=False)


object_ids = []
for i, l2_factor in enumerate([1e-5, 1e-4, 1e-3]):
    for learning_rate in [1e-2, 1e-1, 1]:
        object_ids += [keras_example_main.remote(
            200, 8, l2_factor, learning_rate,
            experiment_dir=(
                './experiments/ray_modelV1_lr[%s]_l2[%s]'
                % (learning_rate, l2_factor)))]
        print(object_ids[-1])


# To make sure that the ray stays alive until all processes are finished.
ray.get(object_ids)
