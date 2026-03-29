import pandas as pd
from .file_handler import FileHandler
from .eda import EDA

class Preprocessor:
    def __init__(self):
        pass

    def drop_missing_from_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.dropna(subset=cols)
        return df
    
    def drop_duplicates_from_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.drop_duplicates(subset=cols, keep="first")
        return df
    
    def clean_data_types_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["user_id"] = df["user_id"].astype(int)
        df["recipe_id"] = df["recipe_id"].astype(int)
        df["rating"] = df["rating"].astype(float)
        return df
    
    def clean_data_types_recipes(self, df: pd.DataFrame) -> pd.DataFrame:
        df["id"] = df["id"].astype(int)
        df["contributor_id"] = df["contributor_id"].astype(int)
        return df
    
    def clean_text(self, df: pd.DataFrame, cols_to_clean: list) -> pd.DataFrame:
        for col in cols_to_clean:
            df[col] = df[col].str.lower()
            df[col] = df[col].str.replace(r"[^\w\s$€]", " ", regex=True)
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        return df

# df_interactions = FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv")
# df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
# data_types_recipes, _ = EDA().data_type_report(df=df_recipes)


# data_types_interactions, _ = EDA().data_type_report(df=df_interactions)

# df_recipes = Preprocessor().drop_missing_from_df(df=df_recipes, cols=["id"])
# df_recipes = Preprocessor().clean_data_types_recipes(df=df_recipes)
# print(len(df_recipes))
# df_recipes = Preprocessor().drop_duplicates_from_df(df=df_recipes, cols=["id"])
# print(len(df_recipes))
# df_interactions = Preprocessor().drop_missing_from_df(df=df_interactions, cols=["user_id", "recipe_id", "rating"])
# df_interactions = Preprocessor().clean_data_types_interactions(df=df_interactions)
# print(len(df_interactions))
# df_interactions = Preprocessor().drop_duplicates_from_df(df=df_interactions, cols=["user_id", "recipe_id"])
# print(len(df_interactions))