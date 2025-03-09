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
├── 📄 main.py
└── 📄 tuning.py
```

- `pipeline` is a custom package for handling the different components of model training.

    - `pipeline/dataset/raw/` is the directory for placing raw data files.

    - `pipeline/dataset/raw/sample_data.csv` contains a sample PSE data.

    - `pipeline/data.py` contains data-related classes and functions.

    - `pipeline/model.py` contains model-related classes and functions.

    - `pipeline/train.py` contains training-related functions.

    - `pipeline/utils.py` contains additional utility classes and functions.

- `main.py` contains the code for training the model.

- `tuning.py` contains the code for hyperparameter tuning.