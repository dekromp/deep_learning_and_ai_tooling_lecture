#!/bin/bash
# Run tensorflow model server and serve our model.
sudo docker run -p 8501:8501 -v $(pwd)/production_models:/models/ -e MODEL_NAME=keras_example --rm -t tensorflow/serving
