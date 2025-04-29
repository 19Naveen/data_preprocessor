import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class Imputer:
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent', datetime_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.strategy = dict()

    def fit(self, df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = SimpleImputer(strategy=self.numeric_strategy)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                imputer = SimpleImputer(strategy=self.categorical_strategy)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Convert to object temporarily for imputation
                imputer = SimpleImputer(strategy=self.datetime_strategy)
            else:
                print(f"Warning: Column '{col}' has unsupported data type {df[col].dtype}. Dropping.")
                continue

            imputer.fit(df[[col]])
            self.strategy[col] = imputer

    def transform(self, df):
        df_copy = df.copy()
        for col, imputer in self.strategy.items():
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                # Convert to object before transforming datetime columns
                temp_col = df_copy[col].astype('object')
                df_copy[col] = imputer.transform(temp_col.to_frame()).ravel()
                df_copy[col] = pd.to_datetime(df_copy[col])
            else:
                df_copy[col] = imputer.transform(df_copy[[col]]).ravel()
        return df_copy

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

