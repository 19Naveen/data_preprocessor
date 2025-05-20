strategies = {
    "cauchy": {
        "outlier_detection": "MAD",
        "normalization": "log_quantile",
        "imputation": "Winsorized Mean"
    },
    "chi2": {
        "outlier_detection": "IQR",
        "normalization": "boxcox",
        "imputation": "IterativeImputer(estimator=RandomForestRegressor())"
    },
    "expon": {
        "outlier_detection": "Percentile",
        "normalization": "log_powertransformer",
        "imputation": "KNN Imputation"
    },
    "exponpow": {
        "outlier_detection": "Percentile",
        "normalization": "log_powertransformer",
        "imputation": "KNNImputer"
    },
    "gamma": {
        "outlier_detection": "Modified Z-score",
        "normalization": "boxcox",
        "imputation": "Bayesian Imputation (Gamma Prior)"
    },
    "lognorm": {
        "outlier_detection": "Log-space IQR",
        "normalization": "yeojohnson_standard",
        "imputation": "Regression-based Median"
    },
    "norm": {
        "outlier_detection": "Z-score",
        "normalization": "StandardScaler",
        "imputation": "Mean"
    },
    "powerlaw": {
        "outlier_detection": "Percentile",
        "normalization": "log_quantile",
        "imputation": "KNNImputer"
    },
    "rayleigh": {
        "outlier_detection": "Modified Z-score",
        "normalization": "sqrt_reciprocal",
        "imputation": "IterativeImputer(estimator=BayesianRidge())"
    },
    "uniform": {
        "outlier_detection": "Range-based",
        "normalization": "minmaxscalar",
        "imputation": "K-Means Imputation"
    }
}