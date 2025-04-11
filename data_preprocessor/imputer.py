import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class numeric_imputer:
    pass

class categorical_imputer:
    pass

class datetime_imputer:
    pass

class Imputer(numeric_imputer='mean', categorical_imputer='most_frequent', datetime_imputer='most_frequent'):
    def __init__(self, numeric_strategy, categorical_strategy, datetime_strategy):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.strategy = dict()
    
    def fit(self, df):
        for col in df.columns:
        
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = SimpleImputer(strategy = self.numeric_strategy, missing_values=np.nan)
            elif df[col].dtype == 'object':
                imputer = SimpleImputer(strategy = self.categorical_strategy, missing_values=np.nan)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                imputer = SimpleImputer(strategy = self.datetime_strategy, fill_value=pd.NaT, missing_values=np.nan)
            else:
                print(f"Warning: Column {col} has unsupported data type {df[col].dtype} for imputation. dropping.")
                df.drop(columns=[col], inplace=True)
                continue
        
            self.strategy[col] = imputer

    def transform(self, df):
        for col in df.columns:
            if col in self.strategy:
                imputer = self.strategy[col]
                df[col] = imputer.fit_transform(df[[col]])
            else:
                print(f"Warning: Column {col} not found in strategy. Skipping.")
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

'''
most, constant, rf and knn imputation 
🔢 Numerical Imputation Methods
🔹 Basic Methods
Mean Imputation
Replace missing values with the column’s mean.

Median Imputation
Replace missing values with the median — better for skewed distributions.

Mode Imputation
Replace with the most frequent value (rare for numeric, but valid for discretized numbers).

Constant Value Imputation
Use a fixed number (e.g., 0, -999, or a domain-specific placeholder).

🔹 Statistical & Distribution-Based
Random Value from Distribution
Sample from the observed value distribution (uniform, normal, etc.).

Random Sample Imputation
Randomly choose observed values from the same column.

🔹 Window-Based (Local Context)
Rolling Mean / Median Imputation
Use the average (or median) of a rolling window around the missing point.

🔹 Interpolation Techniques
Linear Interpolation
Estimate values linearly between known points.

Polynomial Interpolation
Use higher-degree polynomials to estimate values (can overfit).

Spline Interpolation
Smooth curve-fitting (better for non-linear trends).

Time Interpolation
If indexed by time, interpolate respecting the temporal sequence.

🔹 Model-Based Imputation
Regression Imputation
Predict missing value using a regression model trained on other features.

KNN Imputation
Use the mean of nearest neighbors (based on feature similarity).

Decision Tree Imputation
Predict missing values using decision tree models.

Random Forest Imputation (MissForest)
Iterative imputation using random forests — robust and powerful.

Multivariate Imputation by Chained Equations (MICE)
Model each feature with missing values as a function of the others — highly flexible.

🔹 Advanced / Probabilistic
Expectation-Maximization (EM)
Iteratively estimates missing data using likelihood maximization.

Bayesian Imputation
Impute with distributions drawn from a posterior predictive distribution.

'''