# This file contains model-related classes and functions
from torch import nn

# Define dictionary from string to nn.Modules
ACTIVATION_LAYERS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
}

NORM_LAYERS = {
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
    'none': nn.Identity,
}


# A simple LSTM-based model
class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout, norm, num_layers):
        super(TimeSeriesModel, self).__init__()
        self.temporal_module = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.norm = NORM_LAYERS[norm](hidden_dim)
        self.activation = ACTIVATION_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout)
        self.prediction_module = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_features):
        hidden_features, _ = self.temporal_module(input_features)
        hidden_features = hidden_features[:, -1, :]     # Get the hidden state at the final timestep
        hidden_features = self.norm(hidden_features)
        hidden_features = self.activation(hidden_features)
        hidden_features = self.dropout(hidden_features)
        output_features = self.prediction_module(hidden_features)
        
        return output_features