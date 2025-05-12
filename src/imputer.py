import pandas as pd
import numpy as np
from scipy.stats import mstats, trim_mean
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.cluster import KMeans
from Utilities.logger import setup_logger
from typing import Optional

logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Imputer:
    """
    Imputation utility for pandas DataFrames.

    Supports various numeric imputation strategies and optional per-column overrides.

    Parameters
    ----------
    metadata : dict, optional
        Per-column imputation config.
    method_to_all_numeric : str, default='Mean'
        Default numeric imputation method.
    method_to_all_categorical : str, default='Mode'
        Default categorical imputation method.
    method_maps : dict, optional
        Column-to-method override mapping.

    Methods
    -------
    transform(df: pd.DataFrame) -> pd.DataFrame
        Fills missing values in-place according to configured methods.
    """

    def __init__(self,
                 metadata: Optional[dict] = None,
                 method_to_all_numeric: str = 'Mean',
                 method_to_all_categorical: str = 'Mode',
                 method_maps: dict = {}):
        self.method_map = {
            "Winsorized Mean": "_winsorized_mean",
            "IterativeImputer(estimator=RandomForestRegressor())": "_iterative_rf",
            "KNN Imputation": "_knn",
            "KNNImputer": "_knn",
            "Bayesian Imputation (Gamma Prior)": "_bayesian",
            "Regression-based Median": "_regression_median",
            "Mean": "_mean",
            "IterativeImputer(estimator=BayesianRidge())": "_bayesian",
            "K-Means Imputation": "_kmeans"
        }
        self.metadata = metadata if metadata else {}
        self.method_to_all_numeric = self._check_method(method_to_all_numeric)
        self.method_to_all_categorical = method_to_all_categorical
        self.method_map_to_column = method_maps

    def _mean(self, df: pd.DataFrame, col: str):
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        logger.info(f"Filled missing in '{col}' with Mean: {mean_val}")
        return df

    def _winsorized_mean(self, df: pd.DataFrame, col: str):
        series = df[col]
        wins = mstats.winsorize(series.dropna(), limits=[0.05, 0.05])
        wm = wins.mean()
        df[col].fillna(wm, inplace=True)
        logger.info(f"Filled missing in '{col}' with Winsorized Mean: {wm}")
        return df

    def _iterative_rf(self, df: pd.DataFrame, col: str):
        imp = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)
        arr = imp.fit_transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=[np.number]).columns] = arr
        logger.info(f"Applied IterativeImputer(RandomForestRegressor) on numeric features")
        return df

    def _knn(self, df: pd.DataFrame, col: str):
        imp = KNNImputer()
        arr = imp.fit_transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=[np.number]).columns] = arr
        logger.info(f"Applied KNNImputer on numeric features")
        return df

    def _bayesian(self, df: pd.DataFrame, col: str):
        imp = IterativeImputer(estimator=BayesianRidge(), random_state=0)
        arr = imp.fit_transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=[np.number]).columns] = arr
        logger.info(f"Applied IterativeImputer(BayesianRidge) on numeric features")
        return df

    def _regression_median(self, df: pd.DataFrame, col: str):
        # Simple regression-based imputation using other numeric features
        numeric = df.select_dtypes(include=[np.number])
        if col not in numeric:
            return df
        train = numeric[numeric[col].notna()]
        test = numeric[numeric[col].isna()]
        if test.empty or train.shape[1] < 2:
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            logger.info(f"Filled missing in '{col}' with Median fallback: {med}")
            return df
        X_train = train.drop(columns=[col])
        y_train = train[col]
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(test.drop(columns=[col]))
        df.loc[df[col].isna(), col] = preds
        logger.info(f"Filled missing in '{col}' with Regression-based Median (predictions)")
        return df

    def _kmeans(self, df: pd.DataFrame, col: str):
        numeric = df.select_dtypes(include=[np.number]).copy()
        n_clusters = min(5, len(df))
        km = KMeans(n_clusters=n_clusters, random_state=0)
        filled = numeric.fillna(numeric.mean())
        clusters = km.fit_predict(filled)
        numeric['__cluster'] = clusters
        df['__cluster'] = clusters
        for cluster in np.unique(clusters):
            mask = (df['__cluster'] == cluster)
            cluster_mean = df.loc[mask, col].mean()
            df.loc[(df[col].isna()) & mask, col] = cluster_mean
        df.drop(columns='__cluster', inplace=True)
        logger.info(f"Filled missing in '{col}' using K-Means Imputation with {n_clusters} clusters")
        return df

    def _check_method(self, method: str) -> str:
        if method not in self.method_map:
            logger.warning(f"Unknown method '{method}', defaulting to '{self.method_to_all_numeric}'")
            return self.method_to_all_numeric
        return method


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_details = self.metadata.get('columns', {})
        # Impute numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            method_name = self.method_map_to_column.get(col, None)  
            if not method_name:
                method_name = columns_details.get(col, {}).get('imputation', self.method_to_all_numeric)

            method_name = self._check_method(method_name) 
            logger.info(f"Imputing numeric column '{col}' using method: {method_name}")
            func = getattr(self, self.method_map.get(method_name, None), None)
            if callable(func):
                try:
                    df = func(df, col)
                except Exception as e:
                    logger.error(f"Error imputing column '{col}' using {method_name}: {e}")
            else:
                logger.warning(f"Imputation method '{method_name}' for column '{col}' is invalid.")

        # Impute categorical or non-numeric columns
        for col in df.select_dtypes(exclude=[np.number]).columns:
            is_target = col == self.metadata.get('target_column')
            is_categorical = columns_details.get(col, {}).get('dtype') == 'categorical_columns'
            if is_target or is_categorical:
                mode_series = df[col].mode()
                if not mode_series.empty:
                    mode_value = mode_series.iloc[0]
                    df[col] = df[col].fillna(mode_value)
                    logger.info(f"Filled missing values in '{col}' with mode: {mode_value}")
                else:
                    logger.warning(f"Cannot impute '{col}': No mode found (column might be entirely NaN).")

        logger.info("Imputation completed.")
        return df