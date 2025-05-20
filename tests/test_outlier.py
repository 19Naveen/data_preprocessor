import os
import pandas as pd
import pytest
from LazyPrep.outlier import Outlier
from LazyPrep.loader import Loader

def test_outlier_removal():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    df = Loader(data_path, metadata).transform()
    outlier = Outlier(metadata=metadata)
    df_no_outliers = outlier.transform(df)
    assert isinstance(df_no_outliers, pd.DataFrame)
    # Should not be empty
    assert not df_no_outliers.empty
