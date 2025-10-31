# This file contains additional utility functions
import os
import yaml
import torch
import random
import mlflow
import tempfile
import numpy as np
import datetime as dt
from copy import deepcopy
from torchinfo import summary
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .data import StockDataset
from .model import TimeSeriesModel


# Sets seed to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


# Convert nested dictionary to flattened
def flatten_dict(d, parent_key='', separator='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Convert OmegaConf to dictionary
def to_dict(config):
    return flatten_dict(OmegaConf.to_container(config, resolve=True))


# Convert dictionary to OmegaConf
def to_conf(config):
    return OmegaConf.from_dotlist([f'{k}={v}' for k, v in flatten_dict(config).items()])


# Sanitize exp_name
def sanitize_exp_name(exp_name):
    return exp_name.lower().strip('-').strip('_')


# Training set-up
def train_setup(hyperparams, device):
    # Load dataset    
    train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), hyperparams.data.sequence_length, 
                                 transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                                 'features_fit': True, 'targets_fit': True},
                                 raw_path=hyperparams.data.raw_path, cache_path=hyperparams.data.cache_path, data_file=hyperparams.data.data_file)
    val_dataset = StockDataset(dt.datetime(2021,1,1), dt.datetime(2021,12,31), hyperparams.data.sequence_length,
                               transform_spec={'features': train_dataset.features_transform, 'targets': train_dataset.targets_transform, 
                                               'features_fit': False, 'targets_fit': False},
                               raw_path=hyperparams.data.raw_path, cache_path=hyperparams.data.cache_path, data_file=hyperparams.data.data_file)
    test_dataset = StockDataset(dt.datetime(2022,1,1), dt.datetime(2022,12,31), hyperparams.data.sequence_length,
                                transform_spec={'features': train_dataset.features_transform, 'targets': train_dataset.targets_transform, 
                                                'features_fit': False, 'targets_fit': False},
                                raw_path=hyperparams.data.raw_path, cache_path=hyperparams.data.cache_path, data_file=hyperparams.data.data_file)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.data.batch_size, shuffle=True, num_workers=hyperparams.data.nworkers)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.data.batch_size, shuffle=False, num_workers=hyperparams.data.nworkers)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams.data.batch_size, shuffle=False, num_workers=hyperparams.data.nworkers)

    # Extract input shapes
    input_dim = train_dataset[0][0].shape[-1]
    output_dim = train_dataset[0][1].shape[-1]
    
    # Load model
    model = TimeSeriesModel(input_dim, hyperparams.model.hidden_dim, output_dim, hyperparams.model.activation, 
                            hyperparams.model.dropout, hyperparams.model.norm, hyperparams.model.num_layers).to(device)
    # model = torch.compile(model, mode='default')
    
    # [MLFlow] Log model summary
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, 'model_summary.txt')
        with open(tmp_path, 'w') as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(tmp_path)
    
    # [MLFlow] Log dataset hash
    mlflow.log_artifact(os.path.join(hyperparams.data.raw_path, hyperparams.data.data_file + '.dvc'))
    with open(os.path.join(hyperparams.data.raw_path, hyperparams.data.data_file + '.dvc'), 'r') as f:
        dvc_file_content = f.read() 
        data_hash = yaml.safe_load(dvc_file_content)['outs'][0]['md5']
        mlflow.set_tag('DVC Dataset Hash', data_hash)

    # Return all variables
    datasets = (train_dataset, val_dataset, test_dataset)
    loaders = (train_loader, val_loader, test_loader)
    return datasets, loaders, model


# Main training code
def train(model, train_loader, device, loss_fn, optimizer, scaler, hyperparams, **kwargs):
    model.train()
    
    total_loss, total = 0, 0
    for features, targets in train_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device
        
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=hyperparams.train.use_amp):
            predictions = model(features)           # Compute model predictions
            loss = loss_fn(targets, predictions)    # Compute loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + targets.shape[0]
        total_loss = total_loss + loss.item() * targets.shape[0]
        
    return total_loss / total


# Main evaluation code
@torch.no_grad()
def evaluate(model, data_loader, device, metrics, transform, hyperparams, **kwargs):
    model.eval()
   
    total = 0
    total_metrics = {metric: 0 for metric in metrics.keys()}
    targets_all, predictions_all = [], []
    for features, targets in data_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device

        with torch.amp.autocast(device_type=device.type, enabled=hyperparams.train.use_amp):
            predictions = model(features)   # Compute model predictions
        
        # Compute (inverse) transformations
        targets = transform(targets)
        predictions = transform(predictions)
        
        # Compute metrics
        total = total + targets.shape[0]
        for metric, metric_fn in metrics.items():
            total_metrics[metric] = total_metrics[metric] + metric_fn(targets, predictions).item() * targets.shape[0]
    
        # Save targets and predictions
        targets_all.append(targets)
        predictions_all.append(predictions)
    
    # Average metrics
    for metric in metrics.keys():
        total_metrics[metric] = total_metrics[metric] / total
    
    # Concatenate targets and predictions
    targets_all = torch.cat(targets_all).cpu().numpy()
    predictions_all = torch.cat(predictions_all).cpu().numpy()
    
    return total_metrics, {'targets': targets_all, 'predictions': predictions_all}


# Main training loop
def run(model, train_loader, val_loader, device, loss_fn, metrics, transform, hyperparams, **kwargs):
    scaler = torch.amp.GradScaler(device=device.type, enabled=hyperparams.train.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.train.lr, weight_decay=hyperparams.train.wd)   # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams.train.factor, patience=hyperparams.train.patience, min_lr=hyperparams.train.min_lr)      # Scheduler for lr decay
    
    best_val_metric = float('inf')
    best_model_params = deepcopy(model.state_dict())
    for epoch in range(hyperparams.train.epochs):
        loss = train(model, train_loader, device, loss_fn, optimizer, scaler, hyperparams, **kwargs)
        train_metrics, _ = evaluate(model, train_loader, device, metrics, transform, hyperparams, **kwargs)
        val_metrics, _ = evaluate(model, val_loader, device, metrics, transform, hyperparams, **kwargs)
        scheduler.step(loss)
        
        # Update checkpoint
        if hyperparams.train.val_checkpoint in val_metrics:
            if val_metrics[hyperparams.train.val_checkpoint] < best_val_metric:
                best_val_metric = val_metrics[hyperparams.train.val_checkpoint]
                best_model_params = deepcopy(model.state_dict())
        else:
            best_model_params = deepcopy(model.state_dict())
        
        curr_metrics = ({f'train_{metric}': value for metric, value in train_metrics.items()} |
                        {f'val_{metric}': value for metric, value in val_metrics.items()} |
                        {'best_val_metric': best_val_metric})
        yield epoch, curr_metrics, best_model_params
