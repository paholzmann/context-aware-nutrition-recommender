import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.file_handler import FileHandler
from ..features.recipe_features import RecipeFeatures

class BaselineModel:
    def __init__(self):
        pass

    def recommend_popular(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        df_top_n = df.sort_values(["rating_count"], ascending=False).head(top_n)
        return df_top_n
    
    def recommend_top_rated(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        df_best_rated = df.sort_values(["average_rating", "rating_count"], ascending=False).head(top_n)
        return df_best_rated
    
    def recommend_weighted(self, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        df_best_weighted_rated = df.sort_values(["weighted_rating"], ascending=False).head(top_n)
        return df_best_weighted_rated

# df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
# df_interactions = FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv")
# df_rating_count = RecipeFeatures().rating_count_per_recipe(df_interactions=df_interactions)
# df_weighted_rating = RecipeFeatures().weighted_rating_per_recipe(df=df_rating_count)
# df_filtered = RecipeFeatures().add_rating_filter(df=df_weighted_rating, min_ratings=10)
# df_top_n = BaselineModel().recommend_popular(df=df_filtered)
# print(df_top_n)
# df_best_rated = BaselineModel().recommend_top_rated(df=df_filtered)
# print(df_best_rated)
# df_best_weighted_rated = BaselineModel().recommend_weighted(df=df_filtered)
# print(df_best_weighted_rated)