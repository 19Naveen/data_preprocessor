import os
import pandas as pd
import pytest
from LazyPrep.cleaner import Cleaner
from LazyPrep.loader import Loader

def test_cleaner_removes_nulls():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    df = Loader(data_path, metadata).transform()
    cleaner = Cleaner(metadata, column_threshold=0.8, row_threshold=0.8)
    cleaned_df = cleaner.transform(df)
    assert isinstance(cleaned_df, pd.DataFrame)
    # Should not have all-null columns or rows
    assert not cleaned_df.isnull().all(axis=0).any()
    assert not cleaned_df.isnull().all(axis=1).any()
