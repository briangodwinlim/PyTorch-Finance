import os
import torch
import mlflow
import numpy as np
from torch import nn
import datetime as dt
from pprint import pprint
from torchinfo import summary
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from pipeline.train import run
from pipeline.data import StockDataset
from pipeline.model import TimeSeriesModel
from pipeline.utils import Hyperparameter, set_seed

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    # Specify hyperparameters
    hyperparams = Hyperparameter(
        sequence_length = 5,
        batch_size = 32,
        hidden_dim = 32,
        activation = nn.ReLU(),
        dropout = 0,
        norm = nn.BatchNorm1d,
        num_layers = 1,
        lr = 1e-3,
        wd = 0,
        min_lr = 1e-5,
        factor = 0.5,
        patience = 10,
        epochs = 100,
        
        # Set-up
        nworkers = 1,
        nruns = 10,
        log_every = 20,
        use_amp = True,
    )

    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    
    # [MLFlow] Experiment set-up
    ## Option 1: Run `mlflow server --host 127.0.0.1 --port 8080` (before running this script) and uncomment the code below
    # mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
    ## Option 2: Run `mlflow ui --port 8080` (after running this script) on the current directory 
    mlflow.set_experiment(f'Model Training @ {dt.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}')
    
    # Perform training
    results = {metric: [] for metric in metrics.keys()}
    for i in range(hyperparams.nruns):
        with mlflow.start_run(run_name=f'run_{i}'):
            # [MLFlow] Log hyperparameters
            mlflow.log_params(hyperparams.__dict__)
            
            # [MLFlow] Set tag to current run
            mlflow.set_tag('Training Info', f'Model training with seed = {i}')
            
            # Set seed
            set_seed(i)

            # Load dataset    
            train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), hyperparams.sequence_length, 
                                         transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                                         'features_fit': True, 'targets_fit': True})
            val_dataset = StockDataset(dt.datetime(2021,1,1), dt.datetime(2021,12,31), hyperparams.sequence_length,
                                       transform_spec={'features': train_dataset.features_transform, 'targets': train_dataset.targets_transform, 
                                                       'features_fit': False, 'targets_fit': False})
            test_dataset = StockDataset(dt.datetime(2022,1,1), dt.datetime(2022,12,31), hyperparams.sequence_length,
                                        transform_spec={'features': train_dataset.features_transform, 'targets': train_dataset.targets_transform, 
                                                        'features_fit': False, 'targets_fit': False})
            
            train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.nworkers)
            val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)
            test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)
        
            # Extract input shapes
            input_dim = train_dataset[0][0].shape[-1]
            output_dim = train_dataset[0][1].shape[-1]
            
            # Load model
            model = TimeSeriesModel(input_dim, hyperparams.hidden_dim, output_dim, hyperparams.activation, 
                                    hyperparams.dropout, hyperparams.norm, hyperparams.num_layers).to(device)
            model = torch.compile(model, mode='default')
            
            # [MLFlow] Log model summary
            with open('model_summary.txt', 'w') as f:
                f.write(str(summary(model)))
            mlflow.log_artifact('model_summary.txt')
            os.remove('model_summary.txt')

            # Model training
            test_metrics = run(model, train_loader, val_loader, test_loader, device, loss_fn, metrics, 
                            train_dataset.inverse_transform_targets, hyperparams)

            # [MLFlow] Log model
            sample_features, sample_targets = train_dataset[0]
            signature = mlflow.models.infer_signature(sample_features.numpy(), sample_targets.numpy())
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path='model',
                signature=signature,
            )

            # Save results
            for metric in results.keys():
                results[metric].append(test_metrics[metric])        
        
    # Print results
    print('\nSummary of Results')
    results = {f'test_{metric}': values for metric, values in results.items()}
    pprint(results)
    summarize_results = lambda metrics: {metric: f'{np.mean(values):.6f} ± {np.std(values):.6f}' for metric, values in metrics.items()}    
    pprint(summarize_results(results))
