import os
import pandas as pd
import pytest
from LazyPrep.imputer import Imputer
from LazyPrep.loader import Loader

def test_imputer_fills_missing():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    df = Loader(data_path, metadata).transform()
    imputer = Imputer(metadata=metadata)
    df_imputed = imputer.transform(df)
    assert isinstance(df_imputed, pd.DataFrame)
    # Should not be empty
    assert not df_imputed.empty
