import os
import json
from LazyPrep import Transformer

path = os.path.join('Data', 'weather_classification_data.csv')
pipeline = Transformer(path=path, target_column='WeatherType', config=False)

df = pipeline.transform()
print(df.head())

metadata = pipeline.return_metadata()
print(metadata)
print(df.head())
print(df.info())