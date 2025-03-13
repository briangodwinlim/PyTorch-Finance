import os
import ray
import torch
from torch import nn
import datetime as dt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from ray.tune.search.basic_variant import BasicVariantGenerator
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.schedulers import ASHAScheduler

from pipeline.train import train, evaluate
from pipeline.data import StockDataset
from pipeline.model import TimeSeriesModel
from pipeline.utils import Hyperparameter, set_seed


# Commonly used normalization layers
norm_layers = {
    'bn': nn.BatchNorm1d,
    'ln': nn.LayerNorm,
    'none': nn.Identity,
}


def run(config):    
    hyperparams = Hyperparameter(**config, nworkers=1)
    hyperparams.norm = norm_layers[hyperparams.norm]
    hyperparams.activation = nn.ReLU() if hyperparams.activation == 'relu' else nn.Sigmoid()
    
    # Change directory
    os.chdir(base_path)
    
    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

    # Set seed
    set_seed(0)

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
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.wd)   # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hyperparams.factor, patience=hyperparams.patience, min_lr=hyperparams.min_lr)      # Scheduler for lr decay
    
    for epoch in range(hyperparams.epochs):
        loss = train(model, train_loader, device, loss_fn, optimizer)
        val_metrics, _ = evaluate(model, val_loader, device, metrics, train_dataset.inverse_transform_targets)
        test_metrics, _ = evaluate(model, test_loader, device, metrics, train_dataset.inverse_transform_targets)
        scheduler.step(loss)
        
        # Checkpoint results
        checkpoint = {'loss': loss}
        checkpoint.update({f'val_{metric}': value for metric, value in val_metrics.items()})
        checkpoint.update({f'test_{metric}': value for metric, value in test_metrics.items()})
        ray.train.report(checkpoint)


if __name__ == '__main__':
    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify hyperparameter space
    param_space = {
        'sequence_length': ray.tune.grid_search([5, 10]),
        'batch_size': ray.tune.grid_search([64, 128]),
        'hidden_dim': ray.tune.grid_search([16, 32]),
        'activation': ray.tune.grid_search(['relu', 'sigmoid']),
        'dropout': ray.tune.uniform(0, 1),
        'norm': ray.tune.grid_search(['bn', 'ln', 'none']),
        'num_layers': ray.tune.randint(1, 4),
        'lr': ray.tune.loguniform(1e-4, 1e-2),
        'wd': ray.tune.loguniform(1e-5, 1e-3),
        'min_lr': ray.tune.grid_search([1e-5]),
        'factor': ray.tune.quniform(0, 1, 0.1),
        'patience': ray.tune.grid_search([10]),
        'epochs': ray.tune.grid_search([100]),
    }
    
    # Specify hyperparameters to evaluate
    hyperparams = [
    ]
    
    # Specify directories
    base_path = os.getcwd()
    directory = os.path.join(base_path, 'logs/')
    exp_name = 'tuning'
    
    search_alg = BasicVariantGenerator(points_to_evaluate=hyperparams, max_concurrent=4)
    scheduler = None
    # search_alg = OptunaSearch(points_to_evaluate=hyperparams)
    # search_alg.restore_from_dir(os.path.join(directory, exp_name))
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=6)
    # scheduler = ASHAScheduler(max_t=50, grace_period=50)
    
    if ray.tune.Tuner.can_restore(os.path.join(directory, exp_name)):
        tuner = ray.tune.Tuner.restore(
            path=os.path.join(directory, exp_name), 
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            resume_unfinished=True,
        )
    else:
        tuner = ray.tune.Tuner(
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            tune_config=ray.tune.TuneConfig(mode='min', metric='val_mae', search_alg=search_alg, scheduler=scheduler, num_samples=1),
            run_config=ray.train.RunConfig(name=exp_name, storage_path=directory, failure_config=ray.train.FailureConfig(max_failures=2), 
                                           checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1)),
            param_space=param_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result()

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial test mae: {best_result.metrics["test_mae"]}')
