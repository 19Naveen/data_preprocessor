from src.pipeline import Lazy_Prep

if __name__ == "__main__":  
    import os
    import json

    path = os.path.join('Data', 'weather_classification_data.csv')
    pipeline = Lazy_Prep(path=path, target_column='WeatherType')
    df = pipeline.transform()
    print(df.head())
    
    metadata = pipeline.load_metadata()
    print(metadata)