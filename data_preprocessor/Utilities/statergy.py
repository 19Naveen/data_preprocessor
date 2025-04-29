distribution_strategy = {
    # ✅ Normal Distribution
    "norm": {
        "outlier": "zscore",
        "transform": None,
        "scaler": "standard"
    },

    # ✅ Heavy-tailed Distribution
    "cauchy": {
        "outlier": "mad",
        "transform": None,
        "scaler": "robust"
    },

    # ✅ Chi-squared (Right-skewed, positive only)
    "chi2": {
        "outlier": "iqr",
        "transform": "log",
        "scaler": "robust"
    },

    # ✅ Exponential (Right-skewed)
    "expon": {
        "outlier": "iqr",
        "transform": "log",
        "scaler": "robust"
    },

    # ✅ Exponential Power (can be skewed or heavy-tailed)
    "exponpow": {
        "outlier": "iqr",
        "transform": "yeo-johnson",
        "scaler": "robust"
    },

    # ✅ Gamma (Right-skewed)
    "gamma": {
        "outlier": "iqr",
        "transform": "log",
        "scaler": "robust"
    },

    # ✅ Log-normal (Highly skewed)
    "lognorm": {
        "outlier": "iqr",
        "transform": "log",
        "scaler": "robust"
    },

    # ✅ Power Law (Heavy-tailed, long-tail risk)
    "powerlaw": {
        "outlier": "mad",
        "transform": "log",
        "scaler": "robust"
    },

    # ✅ Rayleigh (Bounded + slightly skewed)
    "rayleigh": {
        "outlier": "quantile",
        "transform": "sqrt",
        "scaler": "minmax"
    },

    # ✅ Uniform (No tails)
    "uniform": {
        "outlier": "bounds",
        "transform": None,
        "scaler": "minmax"
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


        self.df = Outlier(self.metadata).transform(self.df)
        logger.info("Outlier removal Completed...")

        if not self.drop_null:
            self.df = Imputer(self.metadata).transform(self.df, target_column=self.target_column)
            logger.info("Imputation Completed...")

        if self.normalize:
            self.df = Normalizer(self.metadata).normalize(self.df)
            logger.info("Normalization Completed...")