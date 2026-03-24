import pandas as pd

from src.file_handler import FileHandler


class DataHandler:
    def __init__(self):
        pass

    def combine_recipes_and_interactions(self, recipes_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        RAW recipes: name,id,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients
        RAW interactions: user_id,recipe_id,date,rating,review

        matches: id and recipe_id
        """
        return recipes_df.merge(interactions_df, left_on="id", right_on="recipe_id", how="inner")

df_1 = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
df_2 = FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv")
print(DataHandler().combine_recipes_and_interactions(df_1, df_2))