"""Load a keras model and save it in a format understood by tf serving."""
import tensorflow as tf


# Load the keras model from disc.
model = tf.keras.models.load_model(
    './experiments/keras_example/model.h5')

# Set the export path. Tensorflow serving takes the last directory name as the
# version of the model.
export_path = './production_models/keras_example/1'
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# Create a signature definition for tfserving.
# We will use the predict API which allows us to have an arbitrary number of
# inputs and outputs.
model_signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={tensor.name: tf.saved_model.utils.build_tensor_info(tensor)
            for tensor in model.inputs},
    outputs={tensor.name: tf.saved_model.utils.build_tensor_info(tensor)
             for tensor in model.outputs},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

# Serialize the model.
with tf.keras.backend.get_session() as session:
    builder.add_meta_graph_and_variables(
        session,
        [tf.saved_model.tag_constants.SERVING],  # This is just a tag.
        signature_def_map={
            'predict_whatever':
                model_signature,
        })

    # Export the model to the production_models/1 folder.
    builder.save(as_text=True)

# We can inspect the exported model by using the saved_model cli:
# $ saved_model_cli show --dir=./production_models/keras_example1 --all
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

# signature_def['predict_whatever']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['input_x1:0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 10)
#         name: input_x1:0
#     inputs['input_x2:0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 20)
#         name: input_x2:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['output_layer/Sigmoid:0'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 1)
#         name: output_layer/Sigmoid:0
#   Method name is: tensorflow/serving/predict
