import torch
import numpy as np
from torch import nn
import datetime as dt
from pprint import pprint
from torchinfo import summary
from torch.utils.data import DataLoader

from pipeline.train import run
from pipeline.data import StockDataset
from pipeline.model import TimeSeriesModel
from pipeline.utils import Hyperparameter, set_seed


if __name__ == '__main__':
    # Specify hyperparameters
    hyperparams = Hyperparameter(
        sequence_length = 5,
        batch_size = 64,
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
    )

    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    
    
    # Perform training
    results = {metric: [] for metric in metrics.keys()}
    for i in range(hyperparams.nruns):
        # Set seed
        set_seed(i)

        # Load dataset    
        train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), hyperparams.sequence_length)
        val_dataset = StockDataset(dt.datetime(2021,1,1), dt.datetime(2021,12,31), hyperparams.sequence_length)
        test_dataset = StockDataset(dt.datetime(2022,1,1), dt.datetime(2022,12,31), hyperparams.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.nworkers)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)
        test_loader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=False, num_workers=hyperparams.nworkers)
    
        # Extract input shapes
        input_dim = train_dataset[0][0].shape[-1]
        output_dim = train_dataset[0][1].shape[-1]
        
        # Load model
        model = TimeSeriesModel(input_dim, hyperparams.hidden_dim, output_dim, hyperparams.activation, 
                                hyperparams.dropout, hyperparams.norm, hyperparams.num_layers).to(device)
        summary(model)

        # Model training
        test_metrics = run(model, train_loader, val_loader, test_loader, device, loss_fn, metrics, hyperparams)
    
        # Save results
        for metric in results.keys():
            results[metric].append(test_metrics[metric])        
    
    # Print results
    print('\nSummary of Results')
    results = {f'test_{metric}': values for metric, values in results.items()}
    pprint(results)
    summarize_results = lambda metrics: {metric: f'{np.mean(values):.6f} ± {np.std(values):.6f}' for metric, values in metrics.items()}    
    pprint(summarize_results(results))
