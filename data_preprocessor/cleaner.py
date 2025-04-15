import pandas as pd
from Utilities.logger import setup_logger
import logging
import os

# Initialize logger
<<<<<<< HEAD
logger = setup_logger(log_file='pipeline.log', __name__=__name__)
=======
logger = setup_logger(log_file='pipeline.log')
>>>>>>> 9d8a6c61a68ce31dbe5d2535296672cbc0d63661


class Cleaner:
    """
    A class for cleaning and preprocessing pandas DataFrames.
    
    This class provides methods to handle missing values, remove duplicates,
    and prepare data for analysis or modeling.
    
    Parameters:
    -----------
    metadata : dict, optional
        Additional metadata about the dataset
    """
    def __init__(self, metadata=None, drop_null=False):
        """
        Initialize the Cleaner with optional metadata.
        
        Parameters:
        -----------
        metadata : dict, optional
            Additional information about the dataset
        """
        cleaning_stats = {
            'columns_dropped': [],
            'rows_dropped': 0,
            'duplicates_removed': 0
        }
        self.metadata = metadata if metadata is not None else {}
        self.metadata['cleaning_stats'] = self.metadata.get('cleaning_stats', cleaning_stats)

    def _drop_columns_with_many_nulls(self, df, threshold=0.7):
        """
        Drop columns that have more null values than the specified threshold.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame
        threshold : float, default=0.7
            Threshold ratio of null values (0.0 to 1.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with high-null columns removed
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
            
        original_columns = df.columns.tolist()
        null_ratios = df.isnull().mean()
        columns_to_drop = null_ratios[null_ratios > threshold].index.tolist()
        
        if columns_to_drop:
            logger.info(f"Dropping {len(columns_to_drop)} columns with null ratio > {threshold}: {columns_to_drop}")
            self.metadata['cleaning_stats']['columns_dropped'].extend(columns_to_drop)
            df = df.drop(columns=columns_to_drop)
        return df
    
    def _drop_rows_with_many_nulls(self, df, threshold=0.7):
        """
        Drop rows that have more null values than the specified threshold.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame
        threshold : float, default=0.7
            Threshold ratio of null values (0.0 to 1.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with high-null rows removed
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        
        rows_before = len(df)
        row_null_counts = df.isnull().sum(axis=1) / len(df.columns)
        df = df[row_null_counts <= threshold]
        rows_dropped = rows_before - len(df)
        
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with null ratio > {threshold}")
            self.metadata['cleaning_stats']['rows_dropped'] += rows_dropped
        return df
    
    def _drop_rows_with_null_target(self, df, target_column):
        """
        Drop rows where the target column contains null values.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame
        target_column : str
            Name of the target column to check for nulls
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with null target rows removed
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found in DataFrame")
            return df
        
        rows_before = len(df)
        df = df.dropna(subset=[target_column])
        rows_dropped = rows_before - len(df)
        
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with null values in target column '{target_column}'")
            self.metadata['cleaning_stats']['rows_dropped'] += rows_dropped
        return df
    
    def _remove_duplicates(self, df):
        """
        Remove duplicate rows from the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with duplicate rows removed
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        rows_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = rows_before - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            self.metadata['cleaning_stats']['duplicates_removed'] = duplicates_removed
        return df
    
    def get_cleaning_summary(self):
        """
        Get a summary of cleaning operations performed.
        
        Returns:
        --------
        dict
            Dictionary containing cleaning statistics
        """
        return self.metadata.get('cleaning_stats', 'No cleaning operations performed yet.')
    
    def transform(self, df, target_column=None, column_threshold=0.7, row_threshold=0.7):
        """
        Apply the complete data cleaning pipeline to the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame
        target_column : str, optional
            Target column to preserve (rows with null targets will be dropped)
        column_threshold : float, default=0.7
            Threshold for dropping columns with many nulls
        row_threshold : float, default=0.7
            Threshold for dropping rows with many nulls
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        logger.info("-" * 50)
        logger.info(f"Starting data cleaning process on DataFrame with shape: {df.shape}")
        
        df = self._drop_columns_with_many_nulls(df, threshold=column_threshold)
        df = self._drop_rows_with_many_nulls(df, threshold=row_threshold)
        if target_column is not None:
            df = self._drop_rows_with_null_target(df, target_column)
        df = self._remove_duplicates(df)
        
        logger.info(f"Data cleaning completed. Final DataFrame shape: {df.shape}")
        logger.info(f"Cleaning summary: {self.get_cleaning_summary()}")
        return df
    