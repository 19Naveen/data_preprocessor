"""
data_preprocessor
-----------------
A modular preprocessing toolkit for CSV datasets, including:
- Encoding-aware file loading
- Data imputation
- Outlier removal
"""

from .pipeline import preprocess_dataset
from .loader import load_csv_files
from .imputer import apply_imputation
from .cleaner import remove_outliers

__all__ = [
    "preprocess_dataset",
    "load_csv_files",
    "apply_imputation",
    "remove_outliers"
]
