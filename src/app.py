import hydra
import torch
import mlflow
import numpy as np
import pandas as pd
import gradio as gr
import datetime as dt
from sklearn.preprocessing import StandardScaler

from .data import StockDataset


@hydra.main(version_base=None, config_path='../config', config_name='inference')
def main(config):
    # [MLFlow] Inference set-up
    mlflow.set_tracking_uri(uri=config.mlflow.uri)

    # Prepare dataset and model
    dataset = StockDataset(dt.datetime(2018,1,1), dt.datetime(2020,12,31), config.data.sequence_length, 
                           transform_spec={'features': StandardScaler(), 'targets': StandardScaler(), 
                                           'features_fit': True, 'targets_fit': True},
                           raw_path=config.data.raw_path, cache_path=config.data.cache_path, data_file=config.data.data_file)
    model = mlflow.pytorch.load_model(config.mlflow.model_uri)

    # Set-up inference function call
    @torch.no_grad()
    def inference(features):
        model.eval()
        
        if np.where(features == '')[0].size > 0 or features.ndim != 2:
            raise gr.Error('Input must not contain missing values', duration=5, title='Missing values')
        
        features = dataset.features_transform.transform(features)
        features = torch.from_numpy(features).float().unsqueeze(0)
        predictions = model(features).numpy()
        predictions = dataset.targets_transform.inverse_transform(predictions)
        predictions = predictions.squeeze(0).astype(np.float16)
        
        return predictions

    # Set-up file upload function call
    def upload_csv(file):
        df = pd.read_csv(file.name)
        df = df[['AC', 'BPI', 'GTCAP', 'JFC', 'SM']]
        df = df.tail(dataset.sequence_length)
        features = df.to_numpy()
        return features, inference(features)

    # Set-up Gradio interface
    # app = gr.Interface(
    #     fn=inference,
    #     inputs=gr.Dataframe(
    #         headers=['AC', 'BPI', 'GTCAP', 'JFC', 'SM'],
    #         row_count=dataset.sequence_length,
    #         col_count=5,
    #         datatype='number',
    #         type='numpy',
    #         label=f'Input sequential features ({dataset.sequence_length}, 5)',
    #         show_copy_button=True,
    #         show_row_numbers=True,
    #     ),
    #     outputs=gr.Dataframe(
    #         headers=['PSE'],
    #         row_count=1,
    #         col_count=1,
    #         datatype='number',
    #         type='numpy',
    #         label='PSE prediction',
    #         show_copy_button=True,
    #     ),
    #     examples=[
    #         dataset.targets_transform.inverse_transform(dataset[0][0].numpy()).astype(np.float16),
    #         dataset.targets_transform.inverse_transform(dataset[1][0].numpy()).astype(np.float16),
    #     ],
    #     example_labels=['Sample input 1', 'Sample input 2'],
    #     live=False,
    #     title='TimeSeriesModel Inference',
    #     description='A front-end for model inference demonstration',
    #     theme=gr.themes.Soft(),
    # )

    # Set-up Gradio blocks
    with gr.Blocks(theme=gr.themes.Soft(), title=config.app.title) as app:
        gr.HTML(f'<h1 style="text-align: center; margin-bottom: 1rem">{config.app.title}</h1>')
        gr.Markdown('A front-end for model inference demonstration')
        
        with gr.Row():
            with gr.Column(scale=4):
                features=gr.Dataframe(
                    headers=['AC', 'BPI', 'GTCAP', 'JFC', 'SM'],
                    row_count=dataset.sequence_length,
                    col_count=5,
                    datatype='number',
                    type='numpy',
                    label=f'Input sequential features ({dataset.sequence_length}, 5)',
                    show_copy_button=True,
                    show_row_numbers=True,
                )
            with gr.Column(scale=1):
                predictions=gr.Dataframe(
                    headers=['PSE'],
                    row_count=1,
                    col_count=1,
                    datatype='number',
                    type='numpy',
                    label='PSE prediction',
                    show_copy_button=True,
                )
        
        with gr.Row():
            submit = gr.Button(value='Submit', variant='primary', scale=0)
            clear = gr.ClearButton(value='Clear', variant='secondary', scale=0)
            submit.click(fn=inference, inputs=features, outputs=predictions)
            clear.add([features, predictions])
        
        with gr.Row(equal_height=True):
            upload = gr.UploadButton(label='Upload', variant='secondary', icon='https://uxwing.com/wp-content/themes/uxwing/download/web-app-development/upload-icon.svg',
                                     scale=0, type='filepath', file_count='single', file_types=['.csv'])
            upload.upload(fn=upload_csv, inputs=upload, outputs=[features, predictions])
            examples = gr.Examples(
                examples=[
                    dataset.targets_transform.inverse_transform(dataset[0][0].numpy()).astype(np.float16),
                    dataset.targets_transform.inverse_transform(dataset[1][0].numpy()).astype(np.float16),
                ],
                inputs=features,
                outputs=predictions,
                fn=inference,
                label='Sample inputs',
                run_on_click=True,
                example_labels=['Sample input 1', 'Sample input 2'],
            )
            
        app.launch(server_name=config.app.host, server_port=config.app.port)


if __name__ == '__main__':
    main()
