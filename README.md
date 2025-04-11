# `data_preprocessor`

**`data_preprocessor`** is a modular and extensible toolkit for preprocessing structured datasets, purpose-built for efficiency and ease of use. It streamlines critical steps like file loading, missing value imputation, and data cleaning — making it ideal for rapid prototyping, exploratory data analysis, and machine learning pipelines.

---

## 🔧 Key Features

### ✅ Modular Architecture  
Each component is decoupled and independently reusable, allowing you to plug and play as needed.

---

### 📂 Loader
- Seamlessly loads `.csv` and `.xlsx` files.  
- Automatically detects file encoding for robustness.  
- Includes basic validation and error handling to ensure data integrity.

---

### 🔍 Imputer
- Supports multiple strategies to impute missing values.  
- Handles numerical, categorical, and datetime columns.  
- Built for compatibility with ML preprocessing pipelines.

---

### 🧹 Cleaner
- Drops empty columns and rows with all null values.  
- Streamlines essential data cleaning tasks with minimal configuration.

---

### 🧪 Pipeline
- Demonstrates usage by chaining loader, imputer, and cleaner components.  
- Ideal for rapid, real-world preprocessing with minimal boilerplate.

---

## 🚀 Quick Start

    pipeline = Pipeline("DATASET_PATH")
    processed_df = pipeline.run()
    processed_df.head()
