import os
import ray
import hydra
import torch
import mlflow
import numpy as np
from torch import nn
from functools import partial
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient
from ray.tune.search.sample import Domain
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search.basic_variant import BasicVariantGenerator

from .utils import set_seed, train_setup, to_dict, to_conf, sanitize_exp_name, run

OmegaConf.register_new_resolver('sanitize_exp_name', sanitize_exp_name)


def run_template(hyperparams, config, base_path):
    # Change directory
    os.chdir(base_path)
    
    # Specify loss_fn and metrics
    loss_fn = nn.MSELoss()
    metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

    # Specify device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Convert hyperparams to OmegaConf and merge with config
    hyperparams = OmegaConf.merge(config, to_conf(hyperparams))
    
    with mlflow.start_run(run_name=ray.tune.get_context().get_trial_name()):
        # [MLFLow] Log hyperparameters
        mlflow.log_params(to_dict(hyperparams))
        
        # [MLFlow] Set tag to current run
        mlflow.set_tag('Training Info', 'Hyperparameter tuning')
        
        # Set seed
        set_seed(0)

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
            # Log metrics to Ray
            ray.tune.report(curr_metrics)
        
        # [MLFlow] Log model and end run
        model.load_state_dict(best_model_params)
        model.eval()
        sample_features, sample_targets = train_dataset[0]
        sample_features, sample_targets = np.expand_dims(sample_features.numpy(), axis=0), np.expand_dims(sample_targets.numpy(), axis=0)
        signature = mlflow.models.infer_signature(sample_features, sample_targets)
        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            name=ray.tune.get_context().get_trial_name().replace('run', 'model'),
            signature=signature,
            code_paths=[os.path.join(base_path, 'src')],
        )
        mlflow.end_run()


@hydra.main(version_base=None, config_path='../config', config_name='tune')
def main(config):
    # Convert config to OmegaConf
    config = hydra.utils.instantiate(config)
    
    # Specify directories
    base_path = os.getcwd()
    directory = os.path.join(base_path, config.logging.output_dir)
    
    # Partial on run_template
    run = partial(run_template, config=config, base_path=base_path)
    
    # [MLFlow] Experiment set-up
    ## Option 1: Run `mlflow server --host 127.0.0.1 --port 8080` (before running this script)
    if config.logging.mlflow_uri:
        mlflow.set_tracking_uri(uri=config.logging.mlflow_uri)
    ## Option 2: Run `mlflow ui --port 8080` (after running this script) on the current directory
    mlflow.set_experiment(config.logging._exp_name)

    # Specify hyperparameter space
    param_space = {k: v for k, v in to_dict(config).items() if isinstance(v, Domain)}
    
    # Specify hyperparameters to evaluate
    hyperparams = [
    ]
    
    # search_alg = BasicVariantGenerator(points_to_evaluate=hyperparams, max_concurrent=config.tune.max_concurrent)
    # scheduler = None
    search_alg = OptunaSearch(points_to_evaluate=hyperparams)
    # search_alg.restore_from_dir(os.path.join(directory, config.logging._exp_name))
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=config.tune.max_concurrent)
    scheduler = ASHAScheduler(max_t=config.tune.max_t, grace_period=config.tune.grace_period)
    
    if ray.tune.Tuner.can_restore(os.path.join(directory, config.logging._exp_name)):
        tuner = ray.tune.Tuner.restore(
            path=os.path.join(directory, config.logging._exp_name), 
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': config.tune.cpu_count, 'gpu': 1/config.tune.max_concurrent}),
            resume_unfinished=True,
        )
    else:
        trial_name_fn = lambda trial: f'run_{trial.trial_id.split('_')[-1]}'
        tuner = ray.tune.Tuner(
            trainable=ray.tune.with_resources(ray.tune.with_parameters(run), resources={'cpu': config.tune.cpu_count, 'gpu': 1/config.tune.max_concurrent}),
            tune_config=ray.tune.TuneConfig(mode='min', metric='best_val_metric', search_alg=search_alg, scheduler=scheduler, num_samples=config.tune.num_samples,
                                            trial_name_creator=trial_name_fn, trial_dirname_creator=trial_name_fn),
            run_config=ray.tune.RunConfig(name=config.logging._exp_name, storage_path=directory, failure_config=ray.tune.FailureConfig(max_failures=config.tune.max_failures), 
                                           checkpoint_config=ray.tune.CheckpointConfig(num_to_keep=1), verbose=config.tune.verbose),
            param_space=param_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result()
    best_val_metric = best_result.metrics['best_val_metric']

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial val {config.train.val_checkpoint}: {best_val_metric}')
    
    # [MLFlow] Tag best run
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.logging._exp_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.\"mlflow.runName\" = '{best_result.path.split('/')[-1]}'"
    )
    client.set_tag(
        run_id=runs[0].info.run_id,
        key='Run',
        value='Best',
    )

    # Save best config
    best_config = OmegaConf.merge(config, 
                                  to_conf(best_result.config | 
                                          {'logging.exp_name': config.logging.best_exp_name,
                                           'logging._exp_name': config.logging._best_exp_name,
                                           'hydra.run.dir': '${logging.output_dir}/${logging._exp_name}'}))
    best_config_dir = os.path.join(config.logging.output_dir, config.logging._best_exp_name)
    os.makedirs(best_config_dir, exist_ok=True)
    OmegaConf.save(config=best_config, f=os.path.join(best_config_dir, 'config.yaml'))
    

if __name__ == '__main__':
    main()
