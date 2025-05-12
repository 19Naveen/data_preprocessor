import os
import logging
import mimetypes
import chardet
import pandas as pd
from Utilities.logger import setup_logger
logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class Loader:
    """
    Loader is responsible for detecting the file type, encoding (if applicable),
    and loading CSV or Excel files into a pandas DataFrame.
    """
    SUPPORTED_FORMATS = {
        'csv': ['text/csv', '.csv'],
        'xlsx': [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls', '.xlsx'
        ]
    }

    def __init__(self, path: str, metadata: dict = {}):
        """
        Initializes the Loader with a file path.
        Automatically detects the file format and encoding (for CSV).
        """
        logger.info('-'*50)
        logger.info(f"Initializing Loader for path: {path}")
        self.path = path
        self.format = self._detect_file_format()
        self.encoding = self._detect_encoding() if self.format == 'csv' else None
        self.dataframe = None
        self.metadata = metadata if metadata is not None else {}
        result = {'path': self.path,
                'format': self.format,
                'encoding': self.encoding
            }
        self.metadata['file_info'] = self.metadata.get('file_info', result)

    def _detect_file_format(self) -> str:
        """
        Detects the format of the file based on MIME type and extension.

        Returns:
            str: 'csv' or 'xlsx' if recognized.

        Raises:
            ValueError: If the format is unsupported.
        """
        _, ext = os.path.splitext(self.path)
        ext = ext.lower()
        mime_type, _ = mimetypes.guess_type(self.path)

        logger.debug(f"Guessed MIME type: {mime_type}, Extension: {ext}")

        for fmt, identifiers in self.SUPPORTED_FORMATS.items():
            if mime_type in identifiers or ext in identifiers:
                logger.info(f"Detected file format: {fmt}")
                return fmt

        logger.error(f"Unsupported file format for: {self.path}")
        raise ValueError(f"Unsupported or unknown file type: {self.path}")

    def _detect_encoding(self) -> str:
        """
        Detects encoding of the CSV file using chardet.

        Returns:
            str: The detected encoding.

        Raises:
            Exception: If file can't be read or encoding can't be detected.
        """
        try:
            logger.debug("Detecting file encoding...")
            with open(self.path, 'rb') as file:
                raw_data = file.read(4096)
            encoding = chardet.detect(raw_data)['encoding']
            logger.info(f"Detected file encoding: {encoding}")
            return encoding
        except Exception as e:
            logger.exception("Failed to detect encoding.")
            raise

    def _load_csv(self):
        """
        Loads a CSV file into a pandas DataFrame using detected encoding.
        """
        try:
            logger.info(f"Loading CSV file: {self.path}")
            self.dataframe = pd.read_csv(self.path, encoding=self.encoding)
            logger.info("CSV file loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load CSV file.")
            raise

    def _load_xlsx(self):
        """
        Loads an Excel file into a pandas DataFrame.
        """
        try:
            logger.info(f"Loading Excel file: {self.path}")
            self.dataframe = pd.read_excel(self.path)
            logger.info("Excel file loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Excel file.")
            raise
    
    def _normalize_columns(self, df):
        """
        Normalizes column names by converting to lowercase and replacing spaces with underscores.
        """
        def process(name):
            return name.lower().replace(' ', '_')
        return df.rename(columns=process)

    def transform(self) -> pd.DataFrame:
        """
        Main interface method to load the file into a DataFrame.

        Returns:
            pd.DataFrame: Loaded data.

        Raises:
            ValueError: If the file format is not supported.
        """
        loaders = {
            'csv': self._load_csv,
            'xlsx': self._load_xlsx
        }
        logger.info(f"Loading file with format: {self.format}")
        if self.format not in loaders:
            logger.critical(f"Unsupported file format: {self.format}")
            raise ValueError(f"Unsupported file format: {self.format}")

        logger.debug(f"Delegating to loader for format: {self.format}")
        loaders[self.format]()
        result = {'path': self.path,
                'format': self.format,
                'encoding': self.encoding
            }

        logger.debug(f"Loader Summary: {result}")
        # self.dataframe = self._normalize_columns(self.dataframe)
        self.metadata['file_info'] = self.metadata.get('file_info', result)
        return self.dataframe
