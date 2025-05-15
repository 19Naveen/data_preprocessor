import pandas as pd
from Utilities.logger import setup_logger
import datetime

logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class DatePreprocessor:
    def __init__(self, metadata={}, date_time_columns=[], date_time_format=None):
        self.metadata = metadata
        self.date_time_columns = date_time_columns
        self.date_time_format = date_time_format

    def preprocess(self, df):
        pass