# This file contains data-related classes and functions
import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# Sample dataset class to process raw data
class StockDataset(Dataset):
    def __init__(self, start_date, end_date, sequence_length, raw_path='pipeline/dataset/raw/', cache_path='pipeline/dataset/cached/', force_reload=False):
        super(StockDataset, self).__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.raw_path = raw_path
        self.cache_path = cache_path
        
        # Unique identifier for current instance
        cache_file = f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.pkl'
        
        # Process raw data if cached file not available or force_reload
        if force_reload or not os.path.exists(os.path.join(self.cache_path, cache_file)):
            self.features, self.targets = self.process_data()
            os.makedirs(self.cache_path, exist_ok=True)
            with open(os.path.join(self.cache_path, cache_file), 'wb') as file:
                pickle.dump((self.features, self.targets), file)
                
        # Load cached file if available and not force_reload
        else:
            with open(os.path.join(self.cache_path, cache_file), 'rb') as file:
                self.features, self.targets = pickle.load(file) 
        
    # [IMPORTANT] Suppose we want to predict the 1-day PSEi returns based on the returns of a subset of stocks in the past sequence_length trading days
    def process_data(self):
        # Load sample_data.csv which contains data on PSE stocks
        raw_data = pd.read_csv(os.path.join(self.raw_path, 'sample_data.csv'), parse_dates=['dt'], index_col=['dt'])
        
        # Perform data processing
        processed_data = raw_data[['PSE', 'AC', 'BPI', 'GTCAP', 'JFC', 'SM']]           # Select only a subset of stocks
        processed_data = 100 * processed_data.ffill().pct_change().dropna()             # Fill forward missing values, get percent change, and remove nan's
        processed_data = processed_data.loc[self.start_date : self.end_date]            # Select only the desired date range
        
        # Select the features and targets
        features = torch.from_numpy(processed_data[['AC', 'BPI', 'GTCAP', 'JFC', 'SM']].values).float()     # Convert to torch.Tensor
        targets = torch.from_numpy(processed_data[['PSE']].values).float()                                  # Convert to torch.Tensor
        
        return features, targets
    
    # [IMPORTANT] Ensure no data leakage (e.g., only features until time t are used to predict target at time t + 1) 
    def __getitem__(self, i):
        return self.features[i : i + self.sequence_length], self.targets[i + self.sequence_length]
    
    def __len__(self):
        return self.targets.shape[0] - self.sequence_length - 1
