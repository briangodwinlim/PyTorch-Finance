# PyTorch Finance Starter Code

This repository provides a starter code for a finance deep learning project using PyTorch.

The repository is organized as follows.

```
├── 📂 pipeline/
│   ├── 📂 dataset/
│   │   └── 📂 raw/
│   │       └── 📄 sample_data.csv
│   ├── 📄 __init__.py
│   ├── 📄 data.py
│   ├── 📄 model.py
│   ├── 📄 train.py
│   └── 📄 utils.py
├── 📄 .gitignore
├── 📄 requirements.txt
└── 📄 tuning.py
├── 📄 main.py
├── 📄 app.py
└── 📄 serve.py
└── 📄 deploy.py
```

- `pipeline` is a custom package for handling the different components of model training.

    - `pipeline/dataset/raw/` is the directory for placing raw data files.

    - `pipeline/dataset/raw/sample_data.csv` contains a sample PSE data.

    - `pipeline/data.py` contains data-related classes and functions.

    - `pipeline/model.py` contains model-related classes and functions.

    - `pipeline/train.py` contains training-related functions.

    - `pipeline/utils.py` contains additional utility classes and functions.

- `tuning.py` contains the code for hyperparameter tuning.

- `main.py` contains the code for training the model.

- `app.py` contains the code for an inference front-end for the model.

- `serve.py` contains the code for locally serving the model.

- `deploy.py` contains the code for deploying the model to Docker.


## Installation

Install the dependencies with 

```
pip install -r requirements.txt
```