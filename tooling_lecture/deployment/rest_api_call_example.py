"""Example script that shows how the REST-API of the model is called."""
import json

import numpy as np
import requests


url = 'http://localhost:8501/v1/models/keras_example:predict'

# We can inspect the exported model by using the saved_model cli:
# $ saved_model_cli show --dir=./production_models/keras_example1 --all
data = {
    'signature_name': 'predict_whatever',
    'inputs': {
        'input_x1:0': np.random.randn(1, 10).astype(np.float32).tolist(),
        'input_x2:0': np.random.randn(1, 20).astype(np.float32).tolist()
    }}

data = json.dumps(data)
# Send the request to the server.
response = requests.post(url, data=data)

print('The model predicts a value of %.4f.' % response.json()['outputs'][0][0])
