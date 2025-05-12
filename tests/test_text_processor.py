import pandas as pd
from src.text_processor import TextProcessor

if __name__ == "__main__":
    processor = TextProcessor()
    df = pd.read_csv('Data/weather_classification_data.csv')
    processed_data = processor.transform(df, {})
    print(processed_data.head())