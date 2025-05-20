# `lazyprep`

**`lazyprep`** is a modular and extensible Python package for preprocessing structured datasets, purpose-built for efficiency and ease of use. It streamlines critical steps like file loading, missing value imputation, outlier handling, normalization, and data cleaning—making it ideal for rapid prototyping, exploratory data analysis, and machine learning pipelines.

---

## 🔧 Key Features

### ✅ Modular Architecture  
Each component is decoupled and independently reusable, allowing you to plug and play as needed.

---

### 📂 Loader
- Seamlessly loads `.csv` and `.xlsx` files  
- Automatically detects file encoding for robustness  
- Includes basic validation and error handling to ensure data integrity

---

### 🧑‍💻 Analyzer
- Detects data types of columns  
- Identifies column-wise distributions  
- Enables tailored preprocessing based on data characteristics

---

### 🔍 Imputer
- Supports multiple strategies for imputing missing values  
- Handles numerical, categorical, and datetime columns  
- Built for compatibility with ML preprocessing pipelines

---

### ⚠️ Outlier Handler
- Detects and removes outliers based on column distribution  
- Supports multiple strategies (IQR, Z-score, etc.)

---

### 🧹 Cleaner
- Drops empty columns and rows with all null values  
- Simplifies essential cleaning tasks with minimal configuration

---

### 📝 Normalizer
- Normalizes data for ML training  
- Supports column-specific normalization strategies (MinMax, Standard, etc.)

---

### 🧪 Pipeline
- Demonstrates usage by chaining loader, imputer, cleaner, and other components  
- Ideal for real-world preprocessing with minimal boilerplate

---

## 🚀 Quick Start

```python
from lazyprep import Transformer

# Initialize pipeline with dataset path
pipeline = Transformer("path/to/your/dataset.csv")

# Run the full preprocessing pipeline
processed_df = pipeline.run()

# Preview the processed data
print(processed_df.head())
```

---

## 📦 Installation

Install lazyprep from PyPI:

```bash
pip install lazyprep
```

---

## 🧩 Components

- `lazyprep.loader` – File loading utilities
- `lazyprep.analyzer` – Data type and distribution analysis
- `lazyprep.imputer` – Missing value imputation
- `lazyprep.outlier` – Outlier detection and removal
- `lazyprep.cleaner` – Data cleaning utilities
- `lazyprep.normalize` – Data normalization
- `lazyprep.pipeline` – End-to-end preprocessing pipeline

---

## 🧪 Testing

To run the test suite:

```bash
pytest
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

- NaveenkumarD (<naveenkumar19082004@gmail.com>)

---

For more details, see the documentation or explore the source code in the `src/LazyPrep/` directory.
