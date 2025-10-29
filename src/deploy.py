import os
import json
import hydra
import mlflow
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from mlflow.models.flavor_backend_registry import get_flavor_backend

from .data import StockDataset


@hydra.main(version_base=None, config_path='../config', config_name='inference')
def main(config):
    # [MLFlow] Inference set-up
    mlflow.set_tracking_uri(uri=config.mlflow.uri)
    
    # Sample input data
    dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), config.data.sequence_length, 
                           transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                           'features_fit': True, 'targets_fit': True},
                           raw_path=config.data.raw_path, cache_path=config.data.cache_path, data_file=config.data.data_file)
    input_data, _ = dataset[0]
    input_data = np.expand_dims(input_data.numpy(), axis=0)


    # Option 1: Generate dockerfile
    get_flavor_backend(config.mlflow.model_uri, docker_build=True, env_manager=config.deploy.env_manager).generate_dockerfile(
        model_uri=config.mlflow.model_uri, 
        output_dir=config.deploy.output_dir, 
        enable_mlserver=config.deploy.enable_mlserver, 
        base_image=config.deploy.base_image,
    )


    # # Option 2: Build docker image and run locally
    # payload = json.dumps({'inputs': input_data.tolist()})
    # print('Run the following command on the terminal: \n')
    # print(f'curl localhost:{config.deploy.port}/invocations -H "Content-Type: application/json" --data \'{payload}\' \n')
    # mlflow.models.build_docker(
    #     model_uri=config.mlflow.model_uri,
    #     name=config.deploy.image_name,
    #     env_manager=config.deploy.env_manager,
    #     enable_mlserver=config.deploy.enable_mlserver,
    #     base_image=config.deploy.base_image,
    # )
    # os.system(f'docker run --rm --gpus all --publish {config.deploy.port}:8000 --env DISABLE_NGINX=true --env UVICORN_HOST=0.0.0.0 \"{config.deploy.image_name}\"')


if __name__ == '__main__':
    main()
