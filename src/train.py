import os
import hydra
import torch
import mlflow
import tempfile
import numpy as np
import pandas as pd
from torch import nn
from dvclive import Live
from pprint import pprint
import plotly.express as px
from omegaconf import OmegaConf
import plotly.graph_objects as go

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
    results = {metric: [] for metric in metrics.keys()} | {'test_df': []}
    for i in range(hyperparams.train.nruns):
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
                tmp_path = os.path.join(tmp_dir, 'test_predictions.csv')
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
            
            # Export model as ONNX
            sample_features, sample_targets = train_dataset[0]
            model_dir = os.path.join(hyperparams.logging.model_dir, hyperparams.logging._exp_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.onnx.export(
                model=model,
                args=(sample_features.unsqueeze(dim=0)),
                f=os.path.join(model_dir, f'model_{i:05d}.onnx'),
                export_params=True,
                input_names=['features'],
                output_names=['predictions'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'predictions': {0: 'batch_size'},
                },
            )

            # Save metrics
            for metric in results.keys():
                results[metric].append(test_df if metric == 'test_df' else test_metrics[metric])
        
    # Print results
    print('\nSummary of Results')
    test_df_list = results.pop('test_df')
    results = {f'test_{metric}': values for metric, values in results.items()}
    pprint(results)
    summarize_results = lambda metrics: {metric: f'{np.mean(values):.6f} ± {np.std(values):.6f}' for metric, values in metrics.items()}    
    pprint(summarize_results(results))

    # [MLFlow, DVC] Summarize results across nruns
    with (Live(dir=os.path.join(hyperparams.logging.output_dir, hyperparams.logging._exp_name), save_dvc_exp=True) as dvc_live,
          mlflow.start_run(run_name='run_summary')):
        # [DVC] Log hyperparameters and final metrics
        dvc_live.log_params(to_dict(hyperparams))
        for metric, values in results.items():
            dvc_live.log_metric(f'{metric}_mean', np.mean(values))
            dvc_live.log_metric(f'{metric}_std', np.std(values))

        # Aggregate test predictions
        agg_test_df = pd.DataFrame({'dt': test_df_list[0]['dt'],
                                    'targets': test_df_list[0]['targets'],
                                    'mean_predictions': np.mean([df['predictions'] for df in test_df_list], axis=0),
                                    'std_predictions': np.std([df['predictions'] for df in test_df_list], axis=0)})
        agg_test_df['se'] = agg_test_df['std_predictions'] / np.sqrt(hyperparams.train.nruns)
        agg_test_df['upper_bound'] = agg_test_df['mean_predictions'] + agg_test_df['se']
        agg_test_df['lower_bound'] = agg_test_df['mean_predictions'] - agg_test_df['se']
        
        # [MLFlow] Log aggregate test predictions
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'aggregate_test_predictions.csv')
            agg_test_df.set_index('dt').to_csv(tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path='results')
        
        # [MLFlow] Log plot of aggregate test predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name='Lower Bound (Mean - SE)',
            x=agg_test_df['dt'],
            y=agg_test_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            name='Std Error Band',
            x=agg_test_df['dt'],
            y=agg_test_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            name='Actual',
            x=agg_test_df['dt'],
            y=agg_test_df['targets'],
            mode='lines',
            line=dict(color='black', dash='solid'),
        ))
        fig.add_trace(go.Scatter(
            name='Prediction',
            x=agg_test_df['dt'],
            y=agg_test_df['mean_predictions'],
            mode='lines',
            line=dict(color='blue', dash='solid'),
        ))
        fig.update_layout(
            title='Average Model Prediction on Test Dataset',
            xaxis_title='Date',
            yaxis_title='',
            legend_title_text='',
            hovermode='x unified',
            template='plotly_white',
        )
        mlflow.log_figure(fig, 'results/aggregate_test_predictions.html')


if __name__ == '__main__':
    main()
