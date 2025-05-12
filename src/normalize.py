import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    LabelEncoder
)

from Utilities.logger import setup_logger
logger = setup_logger(log_file='normalize.log', __name__=__name__)

class Normalizer:
    def __init__(self, metadata: dict = {}, stratergy: dict = {}, normalize_numeric: str='MinMaxScalar', normalize_categoric: str='OneHotEncoding'):
        self.metadata = metadata
        self.stratergy = stratergy
        self.normalize_numeric = normalize_numeric
        self.normalize_categoric = normalize_categoric

    def text_normalizer(self, df, columns):
        pass

    def apply_transformation(self, X, method):
        """
        Apply the specified transformation to the input DataFrame X.

        Parameters:
        - X: pd.DataFrame
        - method: str, one of the following:
            'yeojohnson_standard'
            'minmaxscalar'
            'boxcox'
            'standardscaler'
            'sqrt_reciprocal'
            'log_quantile'
            'log_powertransformer'

        Returns:
        - Transformed NumPy array
        """
        epsilon = 1e-5  # for stability in sqrt/reciprocal/log

        if method == 'yeojohnson_standard':
            pipeline = Pipeline([
                ('yeo', PowerTransformer(method='yeo-johnson')),
                ('scaler', StandardScaler())
            ])
            return pipeline.fit_transform(X)

        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            return scaler.fit_transform(X)

        elif method == 'boxcox':
            X_pos = X - X.min() + epsilon
            transformer = PowerTransformer(method='box-cox', standardize=False)
            return transformer.fit_transform(X_pos)

        elif method == 'standardscaler':
            scaler = StandardScaler()
            return scaler.fit_transform(X)

        elif method == 'sqrt_reciprocal':
            return 1.0 / (np.sqrt(X + epsilon))

        elif method == 'log_quantile':
            X_log = np.log1p(X)
            transformer = QuantileTransformer(output_distribution='normal', random_state=0)
            return transformer.fit_transform(X_log)

        elif method == 'log_powertransformer':
            X_log = np.log1p(X)
            transformer = PowerTransformer(method='yeo-johnson')
            return transformer.fit_transform(X_log)

        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, df):
        column_details = self.metadata['columns']
        target = self.metadata['target_column']
        
        # NUMERIC TRANSFORMATION
        for column in df.select_dtypes(include=[np.number]).columns:
            if self.normalize_numeric:
                method = self.normalize_numeric
            else:
                method = self.stratergy[column] if (column in self.stratergy) else column_details[column]['normalization']
            df[column] = self.apply_transformation(X=df[column], method=method)

        # CATEGORICAL TRANSFORMATION
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        categorical_columns.pop(target)

        columns_to_transform = []
        text_normalize_column = []
        for column in columns_to_transform:
            if column_details[column]['text_type'] == 'Nominal/Ordinal_Data':
                columns_to_transform.append(column)
            else:
                text_normalize_column.append(column)
        
        # df = text_normalizer(df, text_normalize_column)
        df = pd.get_dummies(df, columns=columns_to_transform)

        # TARGET NORMALIZATION
        label_encoder = LabelEncoder()
        if column_details[target]['dtype'] == 'categorical_columns':
            df[target] = label_encoder.fit_transform(df[target])
            self.labelencoder = label_encoder

        return df