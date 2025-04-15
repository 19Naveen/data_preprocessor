import pandas as pd
from Utilities.logger import setup_logger
logger = setup_logger(log_file='outlier.log', __name__=__name__)

class Outlier:
	"""
	Class to handle outlier detection and removal.
	"""

	def __init__(self, metadata: dict = None):
		self.metadata = metadata if metadata else {}
		
	def _IQR(df):
		numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
		for col in numeric_cols:
			Q1 = df[col].quantile(0.25)
			Q3 = df[col].quantile(0.75)
			IQR = Q3 - Q1
			lower_bound = Q1 - 1.5 * IQR
			upper_bound = Q3 + 1.5 * IQR
			outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
			if not outliers.empty:
				print(f"Warning: Found {len(outliers)} outliers in column '{col}'. Dropping them.")
				df.drop(outliers.index, inplace=True)
			return df

	def _zscore_removal(df):
		numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
		for col in numeric_cols:
			z_scores = (df[col] - df[col].mean()) / df[col].std()
			outliers = df[abs(z_scores) > 3]
			if not outliers.empty:
				print(f"Warning: Found {len(outliers)} outliers in column '{col}'. Dropping them.")
				df.drop(outliers.index, inplace=True)
			return df
		
	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		logger.info("Starting outlier detection and removal...")
		if not self.metadata or 'column_distribution' not in self.metadata:
			logger.warning("No metadata provided for outlier detection. Skipping...")
			return df
		
		method = self.metadata.get('column_distribution')
		for col in df.columns:
			if col in method:
				if method[col]['Distribution'] in ('Normal'):
					df = self._zscore_removal(df)
				if method[col]['Distribution'] in ('Exponential', 'Lognormal'):
					df = self._IQR(df)

		logger.info("Outlier detection and removal completed.")
		return df