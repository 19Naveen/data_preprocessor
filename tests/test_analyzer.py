import json
import pandas as pd
from pathlib import Path
from src.analyzer import Analyzer

if __name__ == "__main__":
    metadata = {}
    data_path = Path('Data/weather_classification_data.csv')
    data = pd.read_csv(data_path)
    analyzer = Analyzer(metadata=metadata)
    metadata = analyzer.analyze(df=data)
    print(json.dumps(metadata, indent=4))