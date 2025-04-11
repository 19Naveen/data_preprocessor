import pandas as pd

class Cleaner:
	def __init__(self):
		pass

	def clean(self, df):
		for col in df.columns:
			if df[col].isnull().all():
				print(f"Warning: Column '{col}' is entirely empty. Dropping this column.")
				df.drop(col, axis=1, inplace=True)
		if df[col].isnull().all(axis=1):
			print(f"Warning: Row with all null values in column '{col}'. Dropping this row.")
			df.drop(df[df[col].isnull()].index, inplace=True)	
