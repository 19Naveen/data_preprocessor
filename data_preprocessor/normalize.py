from Utilities import setup_logger
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = setup_logger(log_file='normalize.log', __name__=__name__)

class Normalizer:
    def __init__(self, metadata: dict = None):
        self.metadata = metadata if metadata else {}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the DataFrame using Min-Max scaling or Standard scaling based on metadata.
        """
        if not self.metadata or 'column_distribution' not in self.metadata:
            logger.warning("No metadata provided for normalization. Skipping...")
            return df

        method = self.metadata.get('column_distribution')
        scaler1 = MinMaxScaler()
        scaler2 = StandardScaler()
        for col in df.columns:
            if col in method:
                if method[col]['distribution'] in ('Near Constant', 'Uniform-like'):
                   df[col] = scaler1.fit_transform(df[col])
                else:                   
                    print(f"Unknown normalization method for column '{col}'. Skipping...")
                    df[col] = scaler1.fit_transform(df[col])

        logger.info("Normalization completed.")
        return df