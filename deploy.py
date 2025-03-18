import os
import json
import mlflow
import numpy as np
import datetime as dt
from pipeline.data import StockDataset
from sklearn.preprocessing import StandardScaler
from mlflow.models.flavor_backend_registry import get_flavor_backend


# [MLFlow] Inference set-up
tracking_uri = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(uri=tracking_uri)
model_uri = 'runs:/123456789/model'     # 'models:/TimeSeriesModel/1'   (After registering the model)

# Deployment options
image_name = 'timeseriesmodel'
env_manager = 'conda'
enable_mlserver = False
base_image = 'ubuntu:20.04'
output_dir = 'docker_deployment'
exposed_port = 5000


# Sample input data
train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), 10, 
                             transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                             'features_fit': True, 'targets_fit': True})
input_data, _ = train_dataset[0]
input_data = np.expand_dims(input_data.numpy(), axis=0)


# Option 1: Generate dockerfile
get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager).generate_dockerfile(
    model_uri=model_uri, 
    output_dir=output_dir, 
    enable_mlserver=enable_mlserver, 
    base_image=base_image,
)


# Option 2: Build docker image and run locally
# payload = json.dumps({"inputs": input_data.tolist()})
# print('Run the following command on the terminal: \n')
# print(f'curl localhost:{exposed_port}/invocations -H "Content-Type: application/json" --data \'{payload}\' \n')
# mlflow.models.build_docker(
#     model_uri=model_uri,
#     name=image_name,
#     env_manager=env_manager,
#     enable_mlserver=enable_mlserver,
#     base_image=base_image,
# )
# os.system(f'docker run --rm --gpus all --publish {exposed_port}:8000 --env DISABLE_NGINX=true \"{image_name}\"')
