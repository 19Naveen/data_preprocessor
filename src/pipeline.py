import os
import json
import logging
import shutil
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from loader import Loader
from cleaner import Cleaner
from analyzer import Analyzer
from outlier import Outlier
from imputer import Imputer
from text_processor import TextProcessor
from Utilities.logger import setup_logger

logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Lazy_Prep:
    """
    Data preprocessing pipeline that encapsulates loading, imputing, and cleaning steps.
    """

    def __init__(self, path: str, target_column: str = '', config: bool = False, config_file: str = None):
        self.metadata: Dict[str, Any] = {}
        self.filepath: str = path
        self.target_column: str = target_column
        self.pipeline: List[Any] = [] 
        self.config_parameters: Dict[str, Dict] = {}
        self.metadata['target_column'] = target_column
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self.load_config_from_file(config_file)
            config = True
            
        if not config:
            self._default_pipeline()

    def add_cleaner(self, column_threshold: float = 0.80, row_threshold: float = 0.80) -> None:
        """
        Add a cleaner component to the pipeline.
        
        Args:
            column_threshold: Threshold for dropping columns with too many missing values
            row_threshold: Threshold for dropping rows with too many missing values
        """
        self.pipeline.append(Cleaner(self.metadata, column_threshold=column_threshold, row_threshold=row_threshold))
        logger.info(f"Cleaner component added with column_threshold: {column_threshold}, row_threshold: {row_threshold}")
    
    def add_outlier_detection(self, method_to_all: str = '', method_map: Dict = {}) -> None:
        """
        Add an outlier detection component to the pipeline.
        
        Args:
            method_to_all: Method to apply to all columns
            method_map: Dictionary mapping columns to specific outlier detection methods
        """
        self.pipeline.append(Outlier(metadata=self.metadata, method_to_all=method_to_all, method_maps=method_map))
        if method_map or method_to_all:
            logger.info(f"Outlier detection method {method_map or method_to_all} added to pipeline.")

    def add_imputer(self, method_to_all_numeric: str = 'Mean',
                   method_to_all_categorical: str = 'Mode',
                   method_map: Dict = {}) -> None:
        """
        Add an imputer component to the pipeline.
        
        Args:
            method_to_all_numeric: Method to apply to all numeric columns
            method_to_all_categorical: Method to apply to all categorical columns
            method_map: Dictionary mapping specific columns to imputation methods
        """
        self.pipeline.append(Imputer(
            metadata=self.metadata, 
            method_to_all_numeric=method_to_all_numeric, 
            method_to_all_categorical=method_to_all_categorical, 
            method_maps=method_map
        ))
        logger.info(f"Imputer component added with method_to_all_numeric: {method_to_all_numeric}, "
                   f"method_to_all_categorical: {method_to_all_categorical}")

    def add_text_processor(self, text_data_columns: List[str] = []) -> None:
        """
        Add a text processor component to the pipeline.
        
        Args:
            text_data_columns: List of columns containing text data to process
        """
        self.pipeline.append(TextProcessor(metadata=self.metadata, text_data_columns=text_data_columns))
        logger.info(f"TextProcessor component added with text_data_columns: {text_data_columns}")

    def add_component(self, component) -> None:
        """
        Add a custom component to the pipeline.
        
        Args:
            component: Custom component that implements a transform method
        
        The component must implement:
            - transform(df): method that accepts and returns a DataFrame
        """
        self.pipeline.append(component)
        logger.info(f"Component {component.__class__.__name__} added to pipeline.")

    def transform(self, **kwargs) -> pd.DataFrame:
        """
        Apply all components in the pipeline to the DataFrame.
        
        Returns:
            Processed DataFrame
        """
        logger.info("==================Processing starts===================")
        df = Loader(self.filepath, self.metadata).transform()
        Analyzer(self.metadata).analyze(df)
        logger.info(f"Initial DataFrame loaded with shape: {df.shape}")
        
        for component in self.pipeline:
            logger.info(f"-----------------Applying component: {component.__class__.__name__}-------------------")
            df = component.transform(df)

        logger.info("==================Processing ends===================")
        return df
    
    def _default_pipeline(self) -> None:
        """Create a default pipeline with standard components."""
        # Clear existing pipeline
        self.pipeline = []
        
        if 'cleaner' in self.config_parameters:
            self.add_cleaner(**self.config_parameters['cleaner'])
        else:
            self.add_cleaner(column_threshold=0.80, row_threshold=0.80)
        
        if 'text_processor' in self.config_parameters:
            self.add_text_processor(**self.config_parameters['text_processor'])
        else:
            self.add_text_processor()
        
        if 'outlier' in self.config_parameters:
            self.add_outlier_detection(**self.config_parameters['outlier'])
        else:
            self.add_outlier_detection()
        
        if 'imputer' in self.config_parameters:
            self.add_imputer(**self.config_parameters['imputer'])
        else:
            self.add_imputer(method_to_all_numeric='Mean', method_to_all_categorical='Mode')
            
        logger.info("Default pipeline created with Cleaner, Outlier, Imputer, and TextProcessor components.")

    def add_configurations(self, cleaner_config: Dict = {}, 
                          outlier_config: Dict = {}, 
                          imputer_config: Dict = {}, 
                          text_processor_config: Dict = {}, 
                          date_config: Dict = {}) -> None:
        """
        Add configurations for the components in the pipeline.
        
        Args:
            cleaner_config: Configuration for the Cleaner component
            outlier_config: Configuration for the Outlier component
            imputer_config: Configuration for the Imputer component
            text_processor_config: Configuration for the TextProcessor component
            date_config: Configuration for date processing
        """
        if cleaner_config:
            self.config_parameters['cleaner'] = cleaner_config
        if outlier_config:
            self.config_parameters['outlier'] = outlier_config
        if imputer_config:
            self.config_parameters['imputer'] = imputer_config
        if text_processor_config:
            self.config_parameters['text_processor'] = text_processor_config
        if date_config:
            self.config_parameters['date'] = date_config
            
        logger.info(f"Configurations added: {self.config_parameters}")
        
        # Rebuild pipeline with new configurations
        self._default_pipeline()
    
    def load_config_from_file(self, config_file: str) -> None:
        """
        Load configurations from a JSON file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.add_configurations(**config)
            logger.info(f"Configurations loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
            raise
    
    def save_config_to_file(self, config_file: str) -> None:
        """
        Save current configurations to a JSON file.
        
        Args:
            config_file: Path to save the configuration
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config_parameters, f, indent=4)
            logger.info(f"Configurations saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {str(e)}")
            raise

    def load_metadata(self) -> str:
        """
        Get the current metadata as a JSON string.
        
        Returns:
            JSON string representation of metadata
        """
        return json.dumps(self.metadata, indent=4) if self.metadata else {}

    def return_pipeline(self) -> None:
        """Print the current components in the pipeline."""
        print("Components in Pipeline:")
        print('-'*25)
        for component in self.pipeline:
            print(component.__class__.__name__)
        print('-'*25)

    def return_logs(self, dest_path: str = '') -> None:
        """
        Copy log file to a destination path.
        
        Args:
            dest_path: Destination path for the log file
        """
        log_path = os.path.join('Logs', 'pipeline.log')
        if os.path.isfile(log_path):
            if os.path.exists(dest_path):
                shutil.copy(log_path, dest_path)
                print(f'Logs copied to {dest_path}')
            else:
                print("ERROR: Destination path doesn't exist")
        else:
            print('No log file created yet... Use the transform function to generate it')


