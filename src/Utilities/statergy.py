strategies = {
    "cauchy": {
        "outlier_detection": "MAD",
        "normalization": "Log + QuantileTransformer",
        "imputation": "Winsorized Mean"
    },
    "chi2": {
        "outlier_detection": "IQR",
        "normalization": "BoxCox (MLE optimized)",
        "imputation": "IterativeImputer(estimator=RandomForestRegressor())"
    },
    "expon": {
        "outlier_detection": "Percentile",
        "normalization": "Log + PowerTransformer",
        "imputation": "KNN Imputation"
    },
    "exponpow": {
        "outlier_detection": "Percentile",
        "normalization": "Log + PowerTransformer",
        "imputation": "KNNImputer"
    },
    "gamma": {
        "outlier_detection": "Modified Z-score",
        "normalization": "BoxCox (MLE optimized)",
        "imputation": "Bayesian Imputation (Gamma Prior)"
    },
    "lognorm": {
        "outlier_detection": "Log-space IQR",
        "normalization": "Yeo-Johnson + StandardScaler",
        "imputation": "Regression-based Median"
    },
    "norm": {
        "outlier_detection": "Z-score",
        "normalization": "StandardScaler",
        "imputation": "Mean"
    },
    "powerlaw": {
        "outlier_detection": "Percentile",
        "normalization": "Log + QuantileTransformer",
        "imputation": "KNNImputer"
    },
    "rayleigh": {
        "outlier_detection": "Modified Z-score",
        "normalization": "Sqrt + Reciprocal Transformation",
        "imputation": "IterativeImputer(estimator=BayesianRidge())"
    },
    "uniform": {
        "outlier_detection": "Range-based",
        "normalization": "MinMaxScaler (Adaptive Range)",
        "imputation": "K-Means Imputation"
    }
}



'''🧰 Outlier Detection Toolkit
Outlier Detection Functions:

scipy.stats.zscore – for standard normal-based detection
IQR Method – using percentiles (Q1, Q3, and IQR rule)
MAD (Median Absolute Deviation) – for heavy-tailed data (e.g., Cauchy, powerlaw)
sklearn.ensemble.IsolationForest – tree-based unsupervised anomaly detection, good for non-linear relationships
sklearn.neighbors.LocalOutlierFactor – density-based method for detecting local anomalies

⚙️ Data Scaling / Normalization Methods
Scalers (from sklearn.preprocessing):

StandardScaler – for Gaussian or symmetric distributions (norm)
MinMaxScaler – for bounded or uniform distributions (uniform, rayleigh)
RobustScaler – for skewed or heavy-tailed data (lognorm, gamma, cauchy, powerlaw, etc.)

🔁 Data Transformation Functions
Transform Utilities:

np.log1p(x) – log transform with +1 offset for skewed positive data (safe for zeros)
np.sqrt(x) – square root transform for moderate skewness
sklearn.preprocessing.power_transform(x, method='yeo-johnson') – handles both positive and negative skew; robust for many shapes
arcsin_sqrt(x) = np.arcsin(np.sqrt(x)) – ideal for proportion/beta-type bounded distributions (e.g., values in [0, 1])
'''

