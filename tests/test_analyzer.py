import os
import pandas as pd
import pytest
from LazyPrep.analyzer import Analyzer
from LazyPrep.loader import Loader

def test_analyzer_analyzes_columns():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'weather_classification_data.csv')
    metadata = {}
    df = Loader(data_path, metadata).transform()
    analyzer = Analyzer(metadata)
    analyzer.analyze(df)
    # After analysis, metadata should have some keys
    assert isinstance(metadata, dict)
    assert len(metadata) > 0
