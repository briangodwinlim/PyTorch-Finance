import torch
import mlflow
import numpy as np
import gradio as gr
import datetime as dt
from pipeline.data import StockDataset
from sklearn.preprocessing import StandardScaler


# [MLFlow] Inference set-up
tracking_uri = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(uri=tracking_uri)
model_uri = 'runs:/123456789/model'     # 'models:/TimeSeriesModel/1'   (After registering the model)

# Prepare models
train_dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), 10, 
                             transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                             'features_fit': True, 'targets_fit': True})
model = mlflow.pytorch.load_model(model_uri)
device = next(model.parameters()).device


# Set-up inference function call
@torch.no_grad()
def inference(features):
    model.eval()
    
    features = train_dataset.features_transform.transform(features)
    features = torch.from_numpy(features).float().unsqueeze(0)
    predictions = model(features.to(device)).cpu().numpy()
    predictions = train_dataset.targets_transform.inverse_transform(predictions)
    predictions = predictions.squeeze(0).astype(np.float16)
    
    return predictions

# Set-up Gradio interface
app = gr.Interface(
    fn=inference,
    inputs=gr.Dataframe(
        headers=['AC', 'BPI', 'GTCAP', 'JFC', 'SM'],
        row_count=train_dataset.sequence_length,
        col_count=5,
        datatype='number',
        type='numpy',
        label=f'Input sequential features ({train_dataset.sequence_length}, 5)',
        show_copy_button=True,
        show_row_numbers=True,
    ),
    outputs=gr.Dataframe(
        headers=['PSE'],
        row_count=1,
        col_count=1,
        datatype='number',
        type='numpy',
        label='PSE prediction',
        show_copy_button=True,
    ),
    examples=[
        train_dataset.targets_transform.inverse_transform(train_dataset[0][0].numpy()).astype(np.float16),
        train_dataset.targets_transform.inverse_transform(train_dataset[1][0].numpy()).astype(np.float16),
    ],
    example_labels=['Sample input 1', 'Sample input 2'],
    live=False,
    title='TimeSeriesModel Inference',
    description='A front-end for model inference demonstration',
    theme=gr.themes.Soft(),
)


if __name__ == '__main__':
    app.launch()