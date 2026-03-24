import pandas as pd

class FileHandler:
    def __init__(self):
        pass

    def csv_to_df(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

# print(FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv"))