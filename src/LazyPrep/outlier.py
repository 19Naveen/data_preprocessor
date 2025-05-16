import pandas as pd
import numpy as np
from Utilities.logger import setup_logger
from typing import Optional
logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Outlier:
	"""
	Outlier detection and removal utility for pandas DataFrames.

	This class supports multiple statistical methods to detect and remove outliers 
	from numeric columns. Methods can be applied globally or on a per-column basis 
	using metadata or a method mapping dictionary.

	Parameters
	----------
	metadata : dict, optional
		Metadata to control outlier handling. Structure:
		{
			"target_column": "target",  # Column to exclude
			"columns": {
				"col1": {"outlier_detection": "Z-score"},
				...
			}
		}
	method_to_all : str, default='IQR'
		Default method to use when no method is specified. Options:
		"IQR", "Z-score", "Percentile", "Modified Z-score", "Range-based", "MAD", "Log-space IQR"
	method_map : dict, optional
		Manual mapping of column names to outlier detection methods.

	Methods
	-------
	transform(df: pd.DataFrame) -> pd.DataFrame
		Applies the specified outlier detection methods and removes outliers in-place.

	Internal Detection Methods
	--------------------------
	_IQR(df, col)                  : Interquartile Range method.
	_zscore(df, col)               : Standard Z-score.
	_percentile(df, col)           : 1st and 99th percentile filter.
	_modified_zscore(df, col)      : Median and MAD-based Z-score.
	_range_based(df, col)          : Removes min and max values.
	_mad(df, col)                  : Median Absolute Deviation method.
	_logspace_IQR(df, col)         : IQR on log-transformed values (positive data only).

	Notes
	-----
	- Only numeric (float64, int64) columns are processed.
	- Non-positive columns are skipped in log-based methods.
	- Logging is handled via `outlier.log`.
	- Modifies the DataFrame in-place.
	"""

	def __init__(self, metadata: Optional[dict] = None, method_to_all='', method_maps={}):
		self.method_map = {
			"IQR": "_IQR",
			"Z-score": "_zscore",
			"Percentile": "_percentile",
			"Modified Z-score": "_modified_zscore",
			"Range-based": "_range_based",
			"MAD": "_mad",
			"Log-space IQR": "_logspace_IQR"  
		}  
		self.metadata = metadata if metadata else {}
		self.method_to_all = self._check_method(method_to_all, key=False)
		self.method_map_to_column = method_maps


	def _IQR(self, df: pd.DataFrame, col: str):
		Q1 = df[col].quantile(0.25)
		Q3 = df[col].quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR
		outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (IQR). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _zscore(self, df: pd.DataFrame, col: str):
		z_scores = (df[col] - df[col].mean()) / df[col].std()
		outliers = df[abs(z_scores) > 3]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (Z-score). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _percentile(self, df: pd.DataFrame, col: str):
		lower = df[col].quantile(0.01)
		upper = df[col].quantile(0.99)
		outliers = df[(df[col] < lower) | (df[col] > upper)]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (Percentile). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _modified_zscore(self, df: pd.DataFrame, col: str):
		median = df[col].median()
		mad = np.median(np.abs(df[col] - median))
		if mad == 0:
			return df
		mzs = 0.6745 * (df[col] - median) / mad #type: ignore
		outliers = df[abs(mzs) > 3.5]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (Modified Z-score). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _range_based(self, df: pd.DataFrame, col: str):
		min_val = df[col].min()
		max_val = df[col].max()
		outliers = df[(df[col] == min_val) | (df[col] == max_val)]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (Range-based). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _mad(self, df: pd.DataFrame, col: str):
		median = df[col].median()
		mad = np.median(np.abs(df[col] - median))
		lower = median - 3 * mad
		upper = median + 3 * mad
		outliers = df[(df[col] < lower) | (df[col] > upper)]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (MAD). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _logspace_IQR(self, df: pd.DataFrame, col: str):
		if (df[col] <= 0).any():
			logger.info(f"Warning: Column '{col}' contains non-positive values. Skipping log-space IQR.")
			return df
		log_col = np.log(df[col])
		Q1 = log_col.quantile(0.25)             #type: ignore
		Q3 = log_col.quantile(0.75)             #type: ignore
		IQR = Q3 - Q1
		lower_bound = np.exp(Q1 - 1.5 * IQR)
		upper_bound = np.exp(Q3 + 1.5 * IQR)
		outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
		if not outliers.empty:
			logger.info(f"Warning: Found {len(outliers)} outliers in column '{col}' (Log-space IQR). Dropping them.")
			df.drop(outliers.index, inplace=True)
		return df

	def _check_method(self, method: str, key: bool = True) -> str:
		if method not in self.method_map:
			method = 'IQR' if not key else self.method_to_all 
		return method
	
	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		columns_details = self.metadata.get('columns', {})
		count = len(df)

		for col in columns_details.keys():
			if col == self.metadata.get('target_column') or columns_details[col]['dtype'] == 'categorical_columns':
				logger.info(f"{col} is {columns_details[col]['dtype']} or Target Column so skipping it")
				continue

			elif self.method_map_to_column and col in self.method_map_to_column:
				method_name = self._check_method(self.method_map_to_column[col])
			else:
				col_meta = columns_details.get(col, {})
				outlier_method = col_meta.get('outlier_detection', self.method_to_all)
				method_name = self._check_method(outlier_method)

			logger.info(f"{col} is {columns_details[col]['dtype']}... Executing {method_name}.....")
			method_func = getattr(self, self.method_map[method_name])
			df = method_func(df, col)
	
		logger.info(f"Outlier detection and removal completed. Total Removed rows {count-len(df)}")
		return df
	


