# This file contains training-related functions
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def train(model, train_loader, device, loss_fn, optimizer, **kwargs):
    model.train()
    
    total_loss, total = 0, 0
    for features, targets in train_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device
        
        optimizer.zero_grad()

        predictions = model(features)           # Compute model predictions
        loss = loss_fn(targets, predictions)    # Compute loss
        
        loss.backward()     # Backpropagation with chain rule
        optimizer.step()    # Gradient descent

        total = total + targets.shape[0]
        total_loss = total_loss + loss.item() * targets.shape[0]
        
    return total_loss / total


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, **kwargs):
    model.eval()
   
    total = 0
    total_metrics = {metric: 0 for metric in metrics.keys()}
    for features, targets in data_loader:
        features = features.to(device)      # Move tensors to device
        targets = targets.to(device)        # Move tensors to device

        predictions = model(features)       # Compute model predictions
        
        # Compute metrics
        total = total + targets.shape[0]
        for metric, metric_fn in metrics.items():
            total_metrics[metric] = total_metrics[metric] + metric_fn(targets, predictions).item() * targets.shape[0]
    
    # Average metrics
    for metric in metrics.keys():
        total_metrics[metric] = total_metrics[metric] / total
    
    return total_metrics


def run(model, train_loader, val_loader, test_loader, device, loss_fn, metrics, hyperparams, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.wd)   # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams.factor, patience=hyperparams.patience, min_lr=hyperparams.min_lr)      # Scheduler for lr decay
    
    for epoch in range(hyperparams.epochs):
        loss = train(model, train_loader, device, loss_fn, optimizer, **kwargs)
        val_metrics = evaluate(model, val_loader, device, metrics, **kwargs)
        test_metrics = evaluate(model, test_loader, device, metrics, **kwargs)
        scheduler.step(loss)
        
        if (epoch + 1) == hyperparams.epochs or (epoch + 1) % hyperparams.log_every == 0:
            print(f'Epoch {epoch+1:04d} | loss: {loss:.6f} | ' + ' | '.join([f'val_{metric}: {value:.4f}' for metric, value in val_metrics.items()]))
            
    return test_metrics
