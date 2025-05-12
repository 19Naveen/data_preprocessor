from src.loader import Loader

if __name__ == "__main__":
    import os
    path = os.path.join('Data', 'weather_classification_data.csv')
    loader = Loader(path=path)
    df = loader.transform()
    print(df.head())
    print(loader.metadata)