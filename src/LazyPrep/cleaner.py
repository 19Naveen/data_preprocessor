import os
import pandas as pd
from typing import Optional

import logging
from Utilities.logger import setup_logger

logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Cleaner:
    """
    A class for cleaning and preprocessing pandas DataFrames.
    Provides methods to handle missing values, remove duplicates,
    and prepare data for analysis or modeling.
    """
    def __init__(
        self,
        metadata: Optional[dict] = {},
        target_column: str = '',
        column_threshold: float = 0.7,
        row_threshold: float = 0.7,
        drop_null: bool = False
    ):
        """
        Initialize the Cleaner with optional metadata.
        """
        cleaning_stats = {
            'columns_dropped': set(),
            'rows_dropped': 0,
            'duplicates_removed': 0
        }
        self.metadata = metadata if metadata is not None else {}
        if 'cleaning_stats' not in self.metadata:
            self.metadata['cleaning_stats'] = cleaning_stats
        else:
            if not isinstance(self.metadata['cleaning_stats'].get('columns_dropped', set()), set):
                self.metadata['cleaning_stats']['columns_dropped'] = set(self.metadata['cleaning_stats']['columns_dropped'])
        self.target_column = self.metadata.get('target_column', None)
        self.column_threshold = column_threshold
        self.row_threshold = row_threshold
        self.drop_null = drop_null

    def _drop_columns_with_many_nulls(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        null_ratios = df.isnull().mean()
        columns_to_drop = null_ratios[null_ratios > threshold].index.tolist()
        if columns_to_drop:
            logger.info(f"Dropping {len(columns_to_drop)} columns with null ratio > {threshold}: {columns_to_drop}")
            self.metadata['cleaning_stats']['columns_dropped'].update(columns_to_drop)
            df = df.drop(columns=columns_to_drop)
            for col in columns_to_drop:
                self.metadata['columns'].pop(col)
        return df

    def _drop_rows_with_many_nulls(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
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

    def _drop_rows_with_null_target(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
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

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        rows_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = rows_before - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            self.metadata['cleaning_stats']['duplicates_removed'] += duplicates_removed
        return df

    def get_cleaning_summary(self) -> dict:
        """
        Get a summary of cleaning operations performed.
        """
        stats = self.metadata.get('cleaning_stats', None)
        if stats and isinstance(stats.get('columns_dropped', None), set):
            stats['columns_dropped'] = list(stats['columns_dropped'])
        return stats if stats else 'No cleaning operations performed yet.'  # type: ignore

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the complete data cleaning pipeline to the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        logger.info(f"Starting data cleaning process on DataFrame with shape: {df.shape}")
        df = self._drop_columns_with_many_nulls(df, threshold=self.column_threshold)
        df = self._drop_rows_with_many_nulls(df, threshold=self.row_threshold)
        if self.target_column is not None:
            df = self._drop_rows_with_null_target(df, self.target_column)
        if self.drop_null:
            df = df.dropna()
            logger.info("Dropped all remaining rows with any null values.")
        df = self._remove_duplicates(df)
        logger.info(f"Data cleaning completed. Final DataFrame shape: {df.shape}")
        logger.info(f"Cleaning summary: {self.get_cleaning_summary()}")
        return df
