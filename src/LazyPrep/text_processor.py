import nltk
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Utilities.logger import setup_logger

# Setup logger
logger = setup_logger(log_file='pipeline.log', __name__=__name__)

class TextProcessor:
    def __init__(self, text_data_columns=[], metadata={}):
        self.metadata = metadata
        self.lemmatizer = WordNetLemmatizer()
        self.text_data_columns = text_data_columns
        self.stop_words = set(stopwords.words('english'))

        if text_data_columns is not None:
            self._download_resources(self)

    def _download_resources(self, text_data_columns):
        nltk.download('punkt', download_dir='../text_processor_cache')
        nltk.download('stopwords', download_dir='../text_processor_cache')
        nltk.download('wordnet', download_dir='../text_processor_cache')



    def preprocess(self, text, text_data=False):
        """Preprocesses the input text by tokenizing, removing stop words, and lemmatizing."""
        text = text.lower().strip()

        if text_data:
            sentences = sent_tokenize(text)
            words = word_tokenize(text) 
            words = [word for word in words if word not in string.punctuation]
            words = [word for word in words if word not in self.stop_words]
            words = [self.lemmatizer.lemmatize(word) for word in words]
            return words
        
        return text


    def transform(self, df):
        """
        Transform function to apply to the DataFrame.
        """
        column_details = self.metadata.get('columns', {})
        for col in df.select_dtypes(exclude=[np.number]).columns:
            text_data = col in self.text_data_columns
            logger.info(f'Proccessing column: {col}')
            df[col] = df[col].apply(lambda x: self.preprocess(x, text_data) if isinstance(x, str) else x)
            if column_details:
                if text_data:
                    column_details[col]['text_type'] = 'text_data'
                else:
                    column_details[col]['text_type'] = 'Nominal/Ordinal_Data'
        return df
        