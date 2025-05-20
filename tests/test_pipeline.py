import os
import pandas as pd
import pytest
from LazyPrep.pipeline import Transformer

def test_transformer_pipeline_runs():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    df = pipeline.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'WeatherType' in df.columns
    assert pipeline.metadata['target_column'] == 'WeatherType'

def test_add_cleaner_and_transform():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.pipeline = []
    pipeline.add_cleaner(column_threshold=0.7, row_threshold=0.7)
    df = pipeline.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.isnull().all(axis=0).any()
    assert not df.isnull().all(axis=1).any()

def test_add_outlier_and_transform():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.pipeline = []
    pipeline.add_outlier_detection(method_to_all='IQR')
    df = pipeline.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_add_imputer_and_transform():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.pipeline = []
    pipeline.add_imputer(method_to_all_numeric='Median', method_to_all_categorical='Mode')
    df = pipeline.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_add_text_processor_and_transform():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.pipeline = []
    pipeline.add_text_processor(text_data_columns=['WeatherType'])
    df = pipeline.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_return_metadata():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.transform()
    metadata = pipeline.return_metadata()
    assert isinstance(metadata, str)
    assert 'target_column' in metadata

def test_return_pipeline(capsys):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    pipeline = Transformer(path=data_path, target_column='WeatherType', config=False)
    pipeline.return_pipeline()
    captured = capsys.readouterr()
    assert 'Components in Pipeline:' in captured.out
