import pandas as pd
from src.cleaner import Cleaner

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    # Assuming the CSV file is located in the 'Data' directory
    df = pd.read_csv('Data/weather_classification_data.csv')
    data = Cleaner().transform(df=df)
    print(data.head())
