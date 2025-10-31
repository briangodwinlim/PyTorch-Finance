# PyTorch Finance Starter Code

This repository provides a starter code for a finance deep learning project using PyTorch.

The repository is organized as follows.

```
├── 📂 config/
│   ├── 📄 dvc_pipeline.yaml
│   ├── 📄 tune.yaml
│   ├── 📄 train.yaml
│   └── 📄 inference.yaml
├── 📂 dataset/
│   └── 📂 raw/
│       └── 📄 sample_data.csv
├── 📂 notebook/
│   ├── 📄 inference.ipynb
├── 📂 src/
│   ├── 📄 __init__.py
│   ├── 📄 data.py
│   ├── 📄 model.py
│   ├── 📄 utils.py
│   ├── 📄 tune.py
│   ├── 📄 train.py
│   ├── 📄 app.py
│   ├── 📄 serve.py
│   └── 📄 deploy.py
├── 📄 dvc.yaml
└── 📄 requirements.txt
```

- `config` contains the default YAML configuration files for the scripts.

    - `config/dvc_pipeline.yaml` contains the hyperparameters and configurations for the DVC pipeline in `dvc.yaml`.

    - `config/tune.yaml` contains the hyperparameters and configurations for `src/tune.py`.
    
    - `config/train.yaml` contains the hyperparameters and configurations for `src/train.py`.

    - `config/inference.yaml` contains the configurations for `notebook/inference.ipynb`, `src/app.py`, `src/serve.py`, and `src/deploy.py`.

- `dataset` is the directory for raw and processed data files (to be tracked by DVC).

    - `dataset/raw/` is the directory for placing raw data files.

    - `dataset/raw/sample_data.csv` contains a sample PSE data.

- `notebook` contains the Jupyter notebooks.

    - `notebook/inference.ipynb` is the notebook for performing model inference and other analysis.

- `src` contains the Python scripts.

    - `src/data.py` contains data-related classes and functions.

    - `src/model.py` contains model-related classes and functions.

    - `src/utils.py` contains additional utility functions.

    - `src/tune.py` contains the code for hyperparameter tuning.

    - `src/train.py` contains the code for training the model.

    - `src/app.py` contains the code for an inference front-end for the model.

    - `src/serve.py` contains the code for locally serving the model.

    - `src/deploy.py` contains the code for deploying the model to Docker.

- `dvc.yaml` defines the DVC pipeline for hyperparameter tuning and model training.

- `requirements.txt` lists the Python package dependencies for this repository.


## Installation

To install the dependencies of this repository, run the command

```
pip install -r requirements.txt
```


## DVC Pipeline

To use the DVC pipeline, first remove the raw dataset from being git-tracked by running the command

```
git rm -r --cached dataset/raw/sample_data.csv
git commit -m 'Stopped Tracking Dataset'
```

To perform hyperparameter tuning and model training using the pipeline, edit the configuration `config/dvc_pipeline.yaml` and then run 

```
dvc exp run
```

This will create `params.yaml` containing the final configurations used in the pipeline.


## Hyperparameter Tuning

To perform hyperparameter tuning, edit the configuration `config/tune.yaml` to specify the search space and then run 

```
python -m src.tune
```


## Model Training

After tuning the hyperparameters, edit the configuration `config/train.yaml` with the best hyperparameter and then run 

```
python -m src.train
```


## Model Inference

To perform model inference, edit the configuration `config/inference.yaml` to specify the MLFlow and other settings. 

Run the notebook `notebook/inference.ipynb` to perform model inference and other analysis.

Run the following to serve the model with a Gradio front-end app

```
python -m src.app
```

Run the following to serve the model locally with an API

```
python -m src.serve
```

Run the following to deploy the model with Docker

```
python -m src.deploy
```
