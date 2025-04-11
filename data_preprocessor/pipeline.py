from data_preprocessor import Loader, Imputer, Cleaner
import pandas as pd

class Pipeline:
    """
    Data preprocessing pipeline that encapsulates loading, imputing, and cleaning steps.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: pd.DataFrame = pd.DataFrame()

    def run(self) -> pd.DataFrame:
        """
        Executes the full pipeline: load -> impute -> clean
        """
        print("[INFO] Starting preprocessing pipeline...")

        # Step 1: Load data
        self.df = Loader(self.filepath).load()
        print("[INFO] Data loaded successfully.")
        print(self.df.head())

        # Step 2: Impute missing values
        self.df = Imputer().fit_transform(self.df)
        print("[INFO] Missing values imputed.")

        # Step 3: Clean data
        self.df = Cleaner().clean(self.df)
        print("[INFO] Data cleaned successfully.")

        print("[INFO] Pipeline execution complete.")
        return self.df


if __name__ == "__main__":
    pipeline = Pipeline("/workspaces/data_preprocessor/Data/weather_classification_data.csv")
    processed_df = pipeline.run()
    print("[INFO] Final processed data:")
    print(processed_df.head())
