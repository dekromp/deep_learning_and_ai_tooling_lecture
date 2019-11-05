"""Example script that shows how the REST-API of the model is called."""
import json
import sys

import numpy as np
import requests


# Url where tensorflow serving is running.
url = 'http://localhost:8501/v1/models/keras_example:predict'

# The data we will send to the server.
data = {
    'signature_name': 'serving_default',
    'inputs': {'input_x': np.random.randn(1, 30).astype(np.float32).tolist()},
}

data = json.dumps(data)
# Send the request to the server.
try:
    response = requests.post(url, data=data)
except requests.exceptions.ConnectionError as e:
    print('The REST-service seems not to be up.')
    sys.exit(1)

print('The model predicts a value of %.4f.' % response.json()['outputs'][0][0])
