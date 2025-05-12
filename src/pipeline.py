import os
import json
import logging
import shutil
import pandas as pd
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

    def __init__(self, path, target_column='', config: bool = False):
        self.metadata: dict = {}
        self.filepath: str = path
        self.target_column: str = target_column
        self.pipeline: list = [] 
        if not config:
            self._default_pipeline()

        self.metadata['target_column'] = target_column

    def add_cleaner(self, column_threshold=0.80, row_threshold=0.80):
        """
        Add a cleaner component to the pipeline.
        """
        self.pipeline.append(Cleaner(self.metadata, column_threshold=column_threshold, row_threshold=row_threshold))
        logger.info(f"Cleaner component added with column_threshold: {column_threshold}, row_threshold: {row_threshold}")
    
    def add_outlier_detection(self, method_to_all='', method_map={}):
        """
        Add an outlier detection component to the pipeline.
        """
        self.pipeline.append(Outlier(metadata=self.metadata, method_to_all=method_to_all, method_maps=method_map))
        if method_map or method_to_all:
            logger.info(f"Outlier detection method {method_map} added to pipeline.")

    def add_imputer(self, method_to_all_numeric='Mean', method_to_all_categorical='Mode', method_map={}):
        """
        Add an imputer component to the pipeline.
        """
        self.pipeline.append(Imputer(metadata=self.metadata, method_to_all_numeric=method_to_all_numeric, method_to_all_categorical=method_to_all_categorical, method_maps=method_map))
        logger.info(f"Imputer component added with method_to_all_numeric: {method_to_all_numeric}, method_to_all_categorical: {method_to_all_categorical}")

    def add_text_processor(self, text_data_columns=[]):
        """
        Add a text processor component to the pipeline.
        """
        self.pipeline.append(TextProcessor(metadata=self.metadata, text_data_columns=text_data_columns))
        logger.info(f"TextProcessor component added with text_data_columns: {text_data_columns}")

    def add_component(self, component):
        """
        Add a custom component to the pipeline.
        Function must accept the following parameters:
        - df: DataFrame to process
        - metadata: Metadata for the DataFrame
        - transform(): Function to apply to the DataFrame

        Function must return the following data:
        - DataFrame: Processed DataFrame
        """
        self.pipeline.append(component)
        logger.info(f"Component {component.__name__} added to pipeline.")

    def transform(self, **kwargs) -> pd.DataFrame:
        """
        Apply all components in the pipeline to the DataFrame.
        """
        logger.info('logger.info(f"=================="Processing starts===================")')
        df = Loader(self.filepath, self.metadata).transform()
        Analyzer(self.metadata).analyze(df)
        logger.info(f"Initial DataFrame loaded with shape: {df.shape}")
        for component in self.pipeline:
            logger.info(f"-----------------Applying component: {component.__class__.__name__}-------------------")
            df = component.transform(df)
            
        logger.info('logger.info(f"==================Processing ends===================")')
        return df
    
    def _default_pipeline(self):
        self.add_cleaner(column_threshold=0.80, row_threshold=0.80)
        self.add_text_processor()
        self.add_outlier_detection()
        self.add_imputer(method_to_all_numeric='Mean', method_to_all_categorical='Mode')
        logger.info("Default pipeline created with Cleaner, Outlier, Imputer, and TextProcessor components.")

    def add_configurations(self, config: dict):
        """
        Add configurations to the pipeline.
        """
        self.metadata.update(config)
        logger.info(f"Configurations added to pipeline: {config}")

    def load_metadata(self):
        """
        Load metadata from a JSON file.
        """
        return json.dumps(self.metadata, indent=4) if self.metadata else {}

    def return_pipeline(self):
        print("Components in Pipeline:")
        print('-'*25)
        for component in self.pipeline:
            print(component.__class__.__name__)
        print('-'*25)

    def return_logs(self, dest_path=''):
        log_path = os.path.join('Logs', 'pipeline.log')
        if os.path.isfile(log_path):
            if os.path.exists(dest_path):
                print(log_path, dest_path)
                shutil.copy(log_path, dest_path)
                print('Logs is copied to the path', dest_path)
            else:
                print("ERROR: Path Doesn't exist")
        else:
            print('No Log File created Yet... Use the transform function to generate it')

if __name__ == "__main__":  
    import os
    import json

    path = os.path.join('Data', 'weather_classification_data.csv')
    pipeline = Lazy_Prep(path=path, target_column='WeatherType', config=False)

    df = pipeline.transform()
    print(df.head())
    
    metadata = pipeline.load_metadata()
    print(metadata)
