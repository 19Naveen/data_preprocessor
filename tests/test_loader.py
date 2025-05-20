import os
import pandas as pd
import pytest
from LazyPrep.loader import Loader

def test_loader_returns_dataframe():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    loader = Loader(data_path, metadata)
    df = loader.transform()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'WeatherType' in df.columns
