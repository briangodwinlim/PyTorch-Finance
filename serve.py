import os
import time
import json
import mlflow
import subprocess
import numpy as np
import datetime as dt
from pipeline.data import StockDataset
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# [MLFlow] Inference set-up
tracking_uri = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(uri=tracking_uri)
os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
model_uri = 'runs:/123456789/model'     # 'models:/TimeSeriesModel/1'   (After registering the model)


# Sample input data
train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), 10, 
                             transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                             'features_fit': True, 'targets_fit': True})
input_data, _ = train_dataset[0]
input_data = np.expand_dims(input_data.numpy(), axis=0)


# Option 1: Python programmatic approach
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager='uv',
)


# Option 2: Set-up an API for the model
# model_api_port = 5050
# process = subprocess.Popen(
#     f'mlflow models serve -m {model_uri} -p {model_api_port} --env-manager uv',
#     shell=True,
#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL,
# )

# time.sleep(5)
# payload = json.dumps({"inputs": input_data.tolist()})
# subprocess.run(
#     f'curl localhost:{model_api_port}/invocations -H "Content-Type: application/json" --data \'{payload}\'',
#     shell=True,
# )

# process.terminate()
# time.sleep(2)
# process.kill()
