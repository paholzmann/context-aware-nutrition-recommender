import pandas as pd
from ..data.file_handler import FileHandler

class RecipeFeatures:
    def __init__(self):
        pass

    def rating_count_per_recipe(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """

        Interactions:
        | recipe_id | rating |
        | 1         | 5      |
        | 1         | 4      |
        | 1         | 5      |

        Output:
        | recipe_id | average_rating | rating_count | 
        | 1         | 4.66           | 3            |
        """
        df_rating_count = df_interactions.groupby("recipe_id")["rating"].agg(["count", "mean"])
        df_rating_count = df_rating_count.rename(columns={"count": "rating_count", "mean": "average_rating"})
        return df_rating_count

    def weighted_rating_per_recipe(self, df: pd.DataFrame, m: int = 10) -> pd.DataFrame:
        """
        Input:
        | recipe_id | average_rating | rating_count | 
        | 1         | 4.66           | 3            |
        | 2         | 5.0            | 7            |

        Output:
        | recipe_id | average_rating | rating_count | average_global_rating | weighted_rating |
        | 1         | 4.66           | 3            | 4.83                  | 
        | 2         | 5.0            | 5            | 4.83                  |
        """
        average_global_rating = df["average_rating"].mean()
        df["weighted_rating"] = df.apply(
            lambda x: x["rating_count"] / (x["rating_count"] + m) * x["average_rating"]
            + m / (x["rating_count"] + m) * average_global_rating,
            axis=1
            )
        return df
    
    def add_rating_filter(self, df: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
        df[f"over_min_rating_{min_ratings}"] = df["rating_count"] >= min_ratings
        return df
    
# df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
# df_interactions = FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv")
# df_rating_count = RecipeFeatures().rating_count_per_recipe(df_interactions=df_interactions)
# df_weighted_rating = RecipeFeatures().weighted_rating_per_recipe(df=df_rating_count)
# df_filtered = RecipeFeatures().add_rating_filter(df=df_weighted_rating, min_ratings=10)
# print(df_filtered)