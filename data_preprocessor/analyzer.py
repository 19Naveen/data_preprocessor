import os
import logging
import pandas as pd
import numpy as np
from fitter import Fitter, get_common_distributions

# Set up logging
os.makedirs('Logs', exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('Logs/pipeline.log')
file_handler.setFormatter(formatter)
if not logger.hasHandlers(): 
    logger.addHandler(file_handler)


class CustomError(Exception):
    pass

class Analyzer:
    def __init__(self, metadata=None):
        self.metadata = metadata

    def analyze_distribution(self, feature: pd.Series) -> dict:
        """
        Analyzes distribution of a numeric pandas Series and recommends preprocessing steps.

        Parameters:
        - feature: pd.Series - numeric column to analyze

        Returns:
        - dict with analysis details
        """
        try:
            fit_sample = feature.sample(min(1500, len(feature)), random_state=42)
            f = Fitter(fit_sample, distributions=get_common_distributions(), timeout=10)
            f.fit()
            best_fit = f.get_best(method='sumsquare_error')
            best_dist_name = list(best_fit.keys())[0]
            best_dist_params = best_fit[best_dist_name]
            
            result = {
                "Distribution": best_dist_name,
                "Parameters": best_dist_params
            }

        except Exception as e:
            logger.warning(f"Fitter failed for '{feature.name}': {str(e)}")
            result = {"Distribution": "Fitter Failed", "Parameters": None}

        logger.info(result)
        self.metadata['column_distribution'][feature.name] = result
        return result


    def column_details(self, df):
        self.metadata['numeric_columns'] = self.metadata.get('numeric_columns', [])
        self.metadata['categorical_columns'] = self.metadata.get('categorical_columns', [])
        self.metadata['datetime_columns'] = self.metadata.get('datetime_columns', [])
        self.metadata['column_distribution'] = self.metadata.get('column_distribution', {})

        for col in df.columns:
            dtype = df[col].dtype
            logger.info(f"Analyzing column: {col} with dtype: {dtype}")
            if pd.api.types.is_numeric_dtype(dtype):
                self.metadata['numeric_columns'].append(col)
                self.metadata['column_distribution'][col] = self.analyze_distribution(df[col])
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.metadata['datetime_columns'].append(col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                self.metadata['categorical_columns'].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                self.metadata['categorical_columns'].append(col)
            else:
                logger.info(f"{col} unknown datatype removing from the dataframe...")
        return self.metadata

    def analyze(self, df):
        try:
            self.column_details(df)
            return True
        except Exception as e:
            logger.exception(f'Error occured during analizing dataframes {e}')
            return False
            

import json
metadata = {
    'numeric_columns': [],
    'categorical_columns': [],
    'datetime_columns': [],
    'column_distribution': {}
}
data = pd.read_csv('Data/weather_classification_data.csv')
analyzer = Analyzer(metadata=metadata).analyze(df=data)

print(json.dumps(metadata, indent=4))
