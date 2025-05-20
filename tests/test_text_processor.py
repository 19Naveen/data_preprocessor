import os
import pandas as pd
import pytest
from LazyPrep.text_processor import TextProcessor
from LazyPrep.loader import Loader

def test_text_processor_runs():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    df = Loader(data_path, metadata).transform()
    text_processor = TextProcessor(metadata=metadata, text_data_columns=[])
    df_processed = text_processor.transform(df)
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed.empty
