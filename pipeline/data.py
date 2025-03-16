# This file contains data-related classes and functions
import os
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset


# Sample dataset class to process raw data
class StockDataset(Dataset):
    def __init__(self, start_date, end_date, sequence_length, transform_spec=dict(), 
                 raw_path='pipeline/dataset/raw/', cache_path='pipeline/dataset/cached/', force_reload=False):
        super(StockDataset, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.transform_spec = transform_spec
        self.raw_path = raw_path
        self.cache_path = cache_path
        
        # Unique identifier for current instance
        cache_file = f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.pkl'
        
        # Process raw data if cached file not available or force_reload
        if force_reload or not os.path.exists(os.path.join(self.cache_path, cache_file)):
            self.features, self.targets, self.features_transform, self.targets_transform = self.process_data()
            os.makedirs(self.cache_path, exist_ok=True)
            with open(os.path.join(self.cache_path, cache_file), 'wb') as file:
                pickle.dump((self.features, self.targets, self.features_transform, self.targets_transform), file)
                
        # Load cached file if available and not force_reload
        else:
            with open(os.path.join(self.cache_path, cache_file), 'rb') as file:
                self.features, self.targets, self.features_transform, self.targets_transform = pickle.load(file) 
        
    # [IMPORTANT] Suppose we want to predict the 1-day ahead PSEi closing price based on the closing price of a subset of stocks in the past sequence_length trading days
    def process_data(self):
        # Load sample_data.csv which contains data on PSE stocks
        raw_data = pd.read_csv(os.path.join(self.raw_path, 'sample_data.csv'), parse_dates=['dt'], index_col=['dt'])
        
        # Perform data processing
        processed_data = raw_data[['PSE', 'AC', 'BPI', 'GTCAP', 'JFC', 'SM']]           # Select only a subset of stocks
        processed_data = processed_data.ffill()                                         # Fill forward missing values
        # processed_data = 100 * processed_data.pct_change().dropna()                   # [NOT USED] Get percent change and remove nan's
        processed_data = processed_data.loc[self.start_date : self.end_date]            # Select only the desired date range
        
        # Select the features and targets
        features = processed_data[['AC', 'BPI', 'GTCAP', 'JFC', 'SM']].values           # Convert to numpy.array
        targets = processed_data[['PSE']].values                                        # Convert to numpy.array
        
        # Transform features
        features_transform = self.transform_spec.get('features', None)                  # features_transform is either None or sklearn object
        if features_transform is not None:
            features_transform = deepcopy(features_transform)
            if self.transform_spec.get('features_fit', True):                           # Fit sklearn object to features
                features_transform = features_transform.fit(features)
            features = features_transform.transform(features)                           # Transform features
        
        # Transform targets
        targets_transform = self.transform_spec.get('targets', None)                    # targets_transform is either None or sklearn object
        if targets_transform is not None:
            targets_transform = deepcopy(targets_transform)
            if self.transform_spec.get('targets_fit', True):                            # Fit sklearn object to targets
                targets_transform = targets_transform.fit(targets)
            targets = targets_transform.transform(targets)                              # Transform targets
        
        # Convert to torch.Tensor
        features = torch.from_numpy(features).float()
        targets = torch.from_numpy(targets).float()
        
        return features, targets, features_transform, targets_transform
    
    # [IMPORTANT] Ensure no data leakage (e.g., only features until time t are used to predict target at time t + 1) 
    def __getitem__(self, i):
        return self.features[i : i + self.sequence_length], self.targets[i + self.sequence_length]
    
    def __len__(self):
        return self.targets.shape[0] - self.sequence_length - 1

    def inverse_transform_features(self, features):
        if self.features_transform is not None:
            device = features.device
            features = features.cpu().numpy()                                   # Convert to numpy.array
            features = self.features_transform.inverse_transform(features)      # Inverse transform features
            features = torch.from_numpy(features).float().to(device)            # Convert to torch.Tensor
        return features
    
    def inverse_transform_targets(self, targets):
        if self.targets_transform is not None:
            device = targets.device
            targets = targets.cpu().numpy()                                     # Convert to numpy.array
            targets = self.targets_transform.inverse_transform(targets)         # Inverse transform targets
            targets = torch.from_numpy(targets).float().to(device)              # Convert to torch.Tensor
        return targets