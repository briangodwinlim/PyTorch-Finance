# This file contains training-related functions
import torch
import mlflow
import numpy as np
from torch import nn
import torch.nn.functional as F


def train(model, train_loader, device, loss_fn, optimizer, scaler, hyperparams, **kwargs):
    model.train()
    
    total_loss, total = 0, 0
    for features, targets in train_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device
        
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=hyperparams.use_amp):
            predictions = model(features)           # Compute model predictions
            loss = loss_fn(targets, predictions)    # Compute loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + targets.shape[0]
        total_loss = total_loss + loss.item() * targets.shape[0]
        
    return total_loss / total


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, transform, hyperparams, **kwargs):
    model.eval()
   
    total = 0
    total_metrics = {metric: 0 for metric in metrics.keys()}
    targets_all, predictions_all = [], []
    for features, targets in data_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device

        with torch.amp.autocast(device_type=device.type, enabled=hyperparams.use_amp):
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


def run(model, train_loader, val_loader, test_loader, device, loss_fn, metrics, transform, hyperparams, **kwargs):
    scaler = torch.amp.GradScaler(device=device.type, enabled=hyperparams.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.wd)   # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams.factor, patience=hyperparams.patience, min_lr=hyperparams.min_lr)      # Scheduler for lr decay
    
    for epoch in range(hyperparams.epochs):
        loss = train(model, train_loader, device, loss_fn, optimizer, scaler, hyperparams, **kwargs)
        val_metrics, _ = evaluate(model, val_loader, device, metrics, transform, hyperparams, **kwargs)
        test_metrics, _ = evaluate(model, test_loader, device, metrics, transform, hyperparams, **kwargs)
        scheduler.step(loss)
        
        # [MLFlow] Log metrics
        current_metrics = {f'val_{metric}': value for metric, value in val_metrics.items()} | {f'test_{metric}': value for metric, value in test_metrics.items()}
        mlflow.log_metrics(current_metrics, step=epoch)
        
        if (epoch + 1) == hyperparams.epochs or (epoch + 1) % hyperparams.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.6f} | ' + ' | '.join([f'val_{metric}: {value:.4f}' for metric, value in val_metrics.items()]))
            
    return test_metrics
