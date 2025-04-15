import json
import logging
import pandas as pd
from loader import Loader
from cleaner import Cleaner
from analyzer import Analyzer
from outlier import Outlier
from imputer import Imputer
from normalize import Normalizer
from Utilities.logger import setup_logger
logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Pipeline:
    """
    Data preprocessing pipeline that encapsulates loading, imputing, and cleaning steps.
    """

    def __init__(self, filepath: str, target_column: str = None, column_threshold: float = 0.7, row_threshold: float = 0.7, normalize: bool = True, drop_null: bool = False):

        self.filepath: str = filepath
        self.target_column: str = target_column
        self.column_threshold: float = column_threshold
        self.row_threshold: float = row_threshold
        self.df: pd.DataFrame = None
        self.metadata: dict = {}
        self.normalize: bool = normalize
        self.drop_null: bool = drop_null

    def load_metadata(self):
        """
        Load metadata from a JSON file.
        """
        return json.dumps(self.metadata, indent=4) if self.metadata else {}
    
    def run(self) -> pd.DataFrame:
        """
        Executes the full pipeline: load -> impute -> clean
        """
        logger.info("Starting preprocessing pipeline...")
        self.df = Loader(self.filepath, self.metadata).load()
        logger.info(self.df.head(3).to_string())

        self.df = Cleaner(self.metadata, self.drop_null).transform(self.df, target_column=self.target_column, column_threshold=self.column_threshold, row_threshold=self.row_threshold)
        logger.info("Preprocessing Completed...")

        Analyzer(self.metadata).analyze(self.df)
        logger.info("Analysis Completed...")

        self.df = Outlier(self.metadata).transform(self.df)
        logger.info("Outlier removal Completed...")

        if not self.drop_null:
            self.df = Imputer(self.metadata).transform(self.df, target_column=self.target_column)
            logger.info("Imputation Completed...")

        if self.normalize:
            self.df = Normalizer(self.metadata).normalize(self.df)
            logger.info("Normalization Completed...")

        logger.info("Pipeline completed successfully.")
        return self.df


pipeline = Pipeline("/workspaces/data_preprocessor/Data/weather_classification_data.csv")
processed_df = pipeline.run()
print(pipeline.load_metadata())