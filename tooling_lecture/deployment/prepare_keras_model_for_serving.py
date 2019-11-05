"""Load a keras model and save it in a format understood by tf serving."""
import os

import tensorflow as tf


this_dir = os.path.dirname(__file__)

# Load the keras model from disc.
model = tf.keras.models.load_model(
    os.path.join(this_dir, 'experiments', 'keras_example', 'model.h5')
)

export_path = os.path.join(this_dir, 'production_models', 'keras_example', '1')

model.save(filepath=export_path, include_optimizer=False, save_format='tf')

"""
We can inspect the exported model by using the saved_model cli:
$ saved_model_cli show --dir=./production_models/keras_example/1 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 30)
        name: serving_default_input_x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_layer'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
"""
