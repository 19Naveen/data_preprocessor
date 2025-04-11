import os
import chardet
import mimetypes
import pandas as pd

class Loader:
    def __init__(self, path):
        self.path = path
        self.format = self.detect_type(path)
        self.encoding = self.detect_encoding(path)
        self.dataframe = None

    def detect_type(self, path):
        mime_type, _ = mimetypes.guess_type(path)
        ext = os.path.splitext(path)[1].lower()

        if mime_type:
            if 'csv' in mime_type:
                return 'csv'
            elif mime_type in [
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ]:
                return 'xlsx'

        # Extension fallback
        if ext == '.csv':
            return 'csv'
        elif ext in ['.xls', '.xlsx']:
            return 'xlsx'

        raise ValueError(f"Unsupported or unknown file type: {path}")


    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)
        return chardet.detect(raw_data)['encoding']

    def load_csv(self, path, encoding):
        self.dataframe = pd.read_csv(path, encoding=encoding)

    def load_xlsx(self, path):
        self.dataframe = pd.read_excel(path)

    def load(self):
        if self.format == 'csv':
            self.load_csv(self.path, self.encoding)
        elif self.format == 'xlsx':
            self.load_xlsx(self.path)
        else:
            raise ValueError(f"Unsupported file format: {self.format}")
        return self.dataframe

