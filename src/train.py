import os
import hydra
import torch
import mlflow
import tempfile
import numpy as np
import pandas as pd
from torch import nn
from pprint import pprint
import plotly.express as px
from omegaconf import OmegaConf

from .utils import set_seed, train_setup, to_dict, sanitize_exp_name, run, evaluate

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

OmegaConf.register_new_resolver('sanitize_exp_name', sanitize_exp_name)


@hydra.main(version_base=None, config_path='../config', config_name='train')
def main(hyperparams):
    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # [MLFlow] Experiment set-up
    ## Option 1: Run `mlflow server --host 127.0.0.1 --port 8080` (before running this script)
    if hyperparams.logging.mlflow_uri:
        mlflow.set_tracking_uri(uri=hyperparams.logging.mlflow_uri)
    ## Option 2: Run `mlflow ui --port 8080` (after running this script) on the current directory 
    mlflow.set_experiment(hyperparams.logging._exp_name)
    
    # Perform training
    results = {metric: [] for metric in metrics.keys()}
    for i in range(hyperparams.logging.nruns):
        with mlflow.start_run(run_name=f'run_{i:05d}'):
            # [MLFlow] Log hyperparameters
            mlflow.log_params(to_dict(hyperparams))
            
            # [MLFlow] Set tag to current run
            mlflow.set_tag('Training Info', f'Model training with seed = {i}')
            
            # Set seed
            set_seed(i)

            # Training set-up
            datasets, loaders, model = train_setup(hyperparams, device)
            train_dataset, val_dataset, test_dataset = datasets
            train_loader, val_loader, test_loader = loaders

            # Model training
            train_step = run(model, train_loader, val_loader, device, loss_fn, metrics, 
                             train_dataset.inverse_transform_targets, hyperparams)
            for epoch, curr_metrics, best_model_params in train_step:
                # [MLFlow] Log metrics
                mlflow.log_metrics(curr_metrics, step=epoch)
                
                # Print metrics
                if (epoch + 1) == hyperparams.train.epochs or (epoch + 1) % hyperparams.logging.log_every == 0:
                    print(f'Epoch {epoch+1:04d} | ' + ' | '.join([f'{metric}: {value:.4f}' for metric, value in curr_metrics.items()]))
           
            # Evaluate on test data using best model parameters
            model.load_state_dict(best_model_params)
            test_metrics, test_outputs = evaluate(model, test_loader, device, metrics, train_dataset.inverse_transform_targets, hyperparams)
            test_df = pd.DataFrame({k: np.squeeze(v) for k, v in test_outputs.items()} | {'dt': test_dataset.index})
            
            # [MLFlow] Log test predictions
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, 'test_predictions.txt')
                test_df.set_index('dt').to_csv(tmp_path)
                mlflow.log_artifact(tmp_path, artifact_path='results')
            fig = px.line(test_df, x='dt', y=['targets', 'predictions'], title='Model Prediction on Test Dataset', template='plotly_white')
            fig.update_traces(mode='lines')
            fig.update_layout(xaxis_title='Date', yaxis_title='', legend_title_text='')
            fig.update_traces(selector=dict(name='targets'), name='Actual', line=dict(color='black', dash='solid'))
            fig.update_traces(selector=dict(name='predictions'), name='Prediction', line=dict(color='blue', dash='solid'))
            mlflow.log_figure(fig, 'results/test_predictions.html')
        
            # [MLFlow] Log model and end run
            model.eval()
            sample_features, sample_targets = train_dataset[0]
            sample_features, sample_targets = np.expand_dims(sample_features.numpy(), axis=0), np.expand_dims(sample_targets.numpy(), axis=0)
            signature = mlflow.models.infer_signature(sample_features, sample_targets)
            mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                name=f'model_{i:05d}',
                signature=signature,
                code_paths=['src'],
            )
            mlflow.end_run()

            # Save results
            for metric in results.keys():
                results[metric].append(test_metrics[metric])        
        
    # Print results
    print('\nSummary of Results')
    results = {f'test_{metric}': values for metric, values in results.items()}
    pprint(results)
    summarize_results = lambda metrics: {metric: f'{np.mean(values):.6f} ± {np.std(values):.6f}' for metric, values in metrics.items()}    
    pprint(summarize_results(results))


if __name__ == '__main__':
    main()
