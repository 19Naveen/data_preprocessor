import os
import json
import pytest
import pandas as pd
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.LazyPrep.pipeline import Lazy_Prep

if __name__ == "__main__":  
    path = os.path.join('Data', 'weather_classification_data.csv')
    pipeline = Lazy_Prep(path=path, target_column='WeatherType')
    df = pipeline.transform()
    df = pipeline.transform()
    print(df.head())
    
    metadata = pipeline.load_metadata()
    print(metadata)

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Temp': [20.5, 19.8, 21.2, 18.4, None],
        'Humidity': [80, 85, None, 75, 90],
        'WindSpeed': [10, 12, 9, None, 11],
        'WeatherType': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy']
    })

@pytest.fixture
def temp_csv(sample_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        sample_data.to_csv(tmp.name, index=False)
        temp_path = tmp.name
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    config = {
        "cleaner_config": {"column_threshold": 0.7, "row_threshold": 0.75},
        "imputer_config": {"method_to_all_numeric": "Median"}
    }
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tmp:
        json.dump(config, tmp)
        temp_path = tmp.name
    yield temp_path
    os.unlink(temp_path)

# Tests
def test_initialization(temp_csv):
    """Test initialization of Lazy_Prep with different parameters."""
    # Basic initialization
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType')
    assert pipeline.filepath == temp_csv
    assert pipeline.target_column == 'WeatherType'
    assert pipeline.metadata['target_column'] == 'WeatherType'
    assert len(pipeline.pipeline) > 0  # Default pipeline should have components

def test_add_components(temp_csv):
    """Test adding different components to the pipeline."""
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType')
    
    # Clear pipeline and add components manually
    pipeline.pipeline = []
    
    # Test adding cleaner
    pipeline.add_cleaner(column_threshold=0.75, row_threshold=0.85)
    assert len(pipeline.pipeline) == 1
    assert pipeline.pipeline[0].__class__.__name__ == 'Cleaner'
    
    # Test adding outlier detection
    pipeline.add_outlier_detection(method_to_all='IQR')
    assert len(pipeline.pipeline) == 2
    assert pipeline.pipeline[1].__class__.__name__ == 'Outlier'
    
    # Test adding imputer
    pipeline.add_imputer(method_to_all_numeric='Median')
    assert len(pipeline.pipeline) == 3
    assert pipeline.pipeline[2].__class__.__name__ == 'Imputer'
    
    # Test adding text processor
    pipeline.add_text_processor(text_data_columns=['WeatherType'])
    assert len(pipeline.pipeline) == 4
    assert pipeline.pipeline[3].__class__.__name__ == 'TextProcessor'

def test_transform(temp_csv, sample_data):
    """Test the transform method."""
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType')
    
    # Mock the components to isolate the transform method test
    with patch('src.pipeline.Loader') as mock_loader:
        mock_loader.return_value.transform.return_value = sample_data
        with patch('src.pipeline.Analyzer') as mock_analyzer:
            # Configure pipeline with simple components that don't change the data
            pipeline.pipeline = [MagicMock(transform=lambda df: df)]
            
            result_df = pipeline.transform()
            
            # Verify transform was called on all components
            assert mock_loader.called
            assert mock_analyzer.called
            assert result_df.equals(sample_data)

def test_config_operations(temp_csv, temp_config_file):
    """Test configuration loading and saving."""
    # Test loading config from file
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType', config=True, config_file=temp_config_file)
    assert 'cleaner' in pipeline.config_parameters
    assert pipeline.config_parameters['cleaner'].get('column_threshold') == 0.7
    
    # Test saving config to file
    new_config_path = f"{temp_config_file}_new"
    pipeline.save_config_to_file(new_config_path)
    assert os.path.exists(new_config_path)
    
    # Clean up
    os.unlink(new_config_path)

def test_metadata_handling(temp_csv):
    """Test metadata handling."""
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType')
    
    # Test loading metadata
    metadata = pipeline.load_metadata()
    assert isinstance(metadata, (dict, str))
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert 'target_column' in metadata
    assert metadata['target_column'] == 'WeatherType'

def test_utility_methods(temp_csv, capsys):
    """Test utility methods like return_pipeline."""
    pipeline = Lazy_Prep(path=temp_csv, target_column='WeatherType')
    
    # Test return_pipeline
    pipeline.return_pipeline()
    captured = capsys.readouterr()
    assert "Components in Pipeline:" in captured.out
    
    # Test add_configurations
    pipeline.add_configurations(
        cleaner_config={"column_threshold": 0.6},
        imputer_config={"method_to_all_numeric": "Median"}
    )
    assert 'cleaner' in pipeline.config_parameters
    assert pipeline.config_parameters['cleaner'].get('column_threshold') == 0.6