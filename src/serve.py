import json
import hydra
import mlflow
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from mlflow.models.flavor_backend_registry import get_flavor_backend

from .data import StockDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


    # Option 1: Python programmatic approach
    mlflow.models.predict(
        model_uri=config.mlflow.model_uri,
        input_data=input_data,
        env_manager=config.serve.env_manager,
    )


    # # Option 2: Set-up an API for the model
    # payload = json.dumps({'inputs': input_data.tolist()})
    # print('Run the following command on the terminal: \n')
    # print(f'curl localhost:{config.serve.api_port}/invocations -H "Content-Type: application/json" --data \'{payload}\' \n')
    # get_flavor_backend(config.mlflow.model_uri, docker_build=True, env_manager=config.serve.env_manager).serve(
    #     model_uri=config.mlflow.model_uri, 
    #     port=config.serve.api_port, 
    #     host=config.serve.api_host,
    #     timeout=config.serve.api_timeout,
    #     enable_mlserver=config.serve.enable_mlserver,
    # )


if __name__ == '__main__':
    main()
