# PyTorch Finance Starter Code

This repository provides a starter code for a finance deep learning project using PyTorch.

The repository is organized as follows.

```
├── 📂 config/
│   ├── 📄 tune.yaml
│   ├── 📄 train.yaml
│   └── 📄 inference.yaml
├── 📂 dataset/
│   └── 📂 raw/
│       └── 📄 sample_data.csv
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
├── 📄 .gitignore
└── 📄 requirements.txt
```

- `config` contains the default YAML configuration files for the scripts.

    - `config/tune.yaml` contains the hyperparameters and configurations for `src/tune.py`.
    
    - `config/train.yaml` contains the hyperparameters and configurations for `src/train.py`.

    - `config/inference.yaml` contains the configurations for `src/app.py`, `src/serve.py`, and `src/deploy.py`.

- `dataset` is the directory for raw and processed data files.

    - `dataset/raw/` is the directory for placing raw data files.

    - `dataset/raw/sample_data.csv` contains a sample PSE data.

- `src` contains the Python scripts.

    - `src/data.py` contains data-related classes and functions.

    - `src/model.py` contains model-related classes and functions.

    - `src/utils.py` contains additional utility functions.

    - `src/tune.py` contains the code for hyperparameter tuning.

    - `src/train.py` contains the code for training the model.

    - `src/app.py` contains the code for an inference front-end for the model.

    - `src/serve.py` contains the code for locally serving the model.

    - `src/deploy.py` contains the code for deploying the model to Docker.


## Installation

To install the dependencies of this repository, run the command

```
pip install -r requirements.txt
```


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

To perform model inference, edit the configuration `config/inference.yaml` to specify the MLFlow settings. 

Run the following to serve the model with a Gradio frontend app

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
