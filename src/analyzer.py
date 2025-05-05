import os
import logging
import pandas as pd
import numpy as np
from fitter import Fitter, get_common_distributions
from pathlib import Path
import json
from Utilities.statergy import strategies

# Set up logging
LOG_DIR = Path('Logs')
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(LOG_DIR / 'pipeline.log')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

class CustomError(Exception):
    pass

class Analyzer:
    def __init__(self, metadata: dict = {}):
        self.metadata = metadata if metadata is not None else {}

    def analyze_distribution(self, feature: pd.Series) -> dict:
        """
        Analyzes distribution of a numeric pandas Series and recommends preprocessing steps.
        """
        try:
            fit_sample = feature.dropna().sample(min(1500, len(feature.dropna())), random_state=42)
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
        logger.info(f"Distribution analysis for {feature.name}: {result}")
        return result

    def column_details(self, df: pd.DataFrame) -> dict:
        self.metadata['columns'] = {}
        for col in df.columns:
            self.metadata['columns'][col] = {}
            dtype = df[col].dtype
            logger.info(f"Analyzing column: {col} with dtype: {dtype}")
            if pd.api.types.is_numeric_dtype(dtype):
                result = self.analyze_distribution(df[col])
                method = result['Distribution']
                self.metadata['columns'][col]['dtype'] = 'numeric_columns'
                self.metadata['columns'][col]['column_distribution'] = method
                self.metadata['columns'][col]['outlier_detection'] = strategies[method].get('outlier_detection', 'IQR')
                self.metadata['columns'][col]['normalization'] = strategies[method].get('normalization', 'MinMaxScaler')
                self.metadata['columns'][col]['imputation'] = strategies[method].get('imputation', 'Mean')

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.metadata['columns'][col]['dtype'] = 'datetime_columns'
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):          #type: ignore
                self.metadata['columns'][col]['dtype'] = 'categorical_columns'
                self.metadata['columns'][col]['unique_values'] = df[col].nunique()
                self.metadata['columns'][col]['top_value'] = df[col].mode()[0] if not df[col].mode().empty else None
            elif pd.api.types.is_bool_dtype(dtype):
                self.metadata['columns'][col]['dtype'] = 'boolean_columns'
            else:
                logger.info(f"{col} unknown datatype, removing from the dataframe...")
        return self.metadata

    def analyze(self, df: pd.DataFrame) -> dict:
        try:
            return self.column_details(df)
        except Exception as e:
            logger.exception(f'Error occurred during analyzing dataframes: {e}')
            raise


