import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.file_handler import FileHandler

class EDA:
    def __init__(self):
        pass

    def get_columns_from_df(self, df: pd.DataFrame) -> list:
        return [col for col in df.columns]
    
    def get_num_columns_rows_from_df(self, df: pd.DataFrame) -> list:
        num_cols = len([col for col in df.columns])
        num_rows = len(df)
        return [num_cols, num_rows]
    
    def missing_values_report(self, df: pd.DataFrame, filepath: str) -> pd.Series:
        missing_by_col = df.isnull().sum()
        missing_by_col.plot(kind="bar")
        plt.title("Missing values report")
        plt.xlabel("Columns")
        plt.ylabel("Number of missing values")
        plt.savefig(filepath)
        return missing_by_col
    
    def check_for_duplicates(self, df: pd.DataFrame) -> bool:
        return df.duplicated().any()
    
    def data_type_report(self, df: pd.DataFrame) -> list:
        data_types = df.dtypes
        data_types_count = data_types.value_counts()
        return [data_types, data_types_count]
    
    def rating_distribution(self, df: pd.DataFrame, filepath: str) -> None:
        plt.figure(figsize=(8, 5))
        sns.countplot(x="rating", data=df)
        plt.title("Rating distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.savefig(filepath)

    def user_interaction_distribution(self, df: pd.DataFrame, filepath: str) -> None:
        user_interactions = df["user_id"].value_counts()
        user_interactions = user_interactions.sort_values()
        cdf = user_interactions.cumsum() / user_interactions.sum()
        plt.figure(figsize=(8,5))
        plt.plot(user_interactions.values, cdf.values)
        plt.xscale("log")
        plt.xlabel("Interactions per user")
        plt.ylabel("Cumulative share of interactions")
        plt.title("User activity CDF")
        plt.grid(True)
        plt.savefig(filepath)

    def recipe_popularity(self, df: pd.DataFrame, filepath: str) -> None:
        _recipe_popularity = df["recipe_id"].value_counts()
        _recipe_popularity = _recipe_popularity.sort_values()
        cdf = _recipe_popularity.cumsum() / _recipe_popularity.sum()
        plt.figure(figsize=(8,5))
        plt.plot(_recipe_popularity.values, cdf.values)
        plt.xscale("log")
        plt.xlabel("Interactions per recipe")
        plt.ylabel("Cumulative share of interactions")
        plt.title("Recipe interaction CDF")
        plt.grid(True)
        plt.savefig(filepath)
    
    def cooking_time_distribution(self, df: pd.DataFrame, filepath: str) -> None:
        cooking_time = df["minutes"].sort_values()
        cdf = np.arange(1, len(cooking_time) + 1) / len(cooking_time)
        plt.figure(figsize=(8, 5))
        plt.plot(cooking_time.values, cdf)
        plt.xscale("log")
        plt.xlabel("Cooking time per recipe")
        plt.ylabel("Cumulative share of cooking time")
        plt.title("Recipe cooking time CDF")
        plt.grid(True)
        plt.savefig(filepath)

    def ingredients_count_distribution(self, df: pd.DataFrame, filepath: str) -> None:
        ingredients_count = df["n_ingredients"].value_counts().sort_index()
        cdf = ingredients_count.cumsum() / ingredients_count.sum()
        plt.figure(figsize=(8,5))
        plt.plot(ingredients_count.index, cdf.values)
        plt.xscale("log")
        plt.xlabel("Ingredients per recipe")
        plt.ylabel("Cumulative share of ingredients per recipe")
        plt.title("Ingredients CDF")
        plt.grid(True)
        plt.savefig(filepath)

    def step_count_distribution(self, df: pd.DataFrame, filepath: str) -> None:
        step_count = df["n_steps"].value_counts().sort_index()
        cdf = step_count.cumsum() / step_count.sum()
        plt.figure(figsize=(8,5))
        plt.plot(step_count.index, cdf.values)
        plt.xscale("log")
        plt.xlabel("Steps per recipe")
        plt.ylabel("Cumulative steps per recipe")
        plt.title("Recipe steps CDF")
        plt.grid(True)
        plt.savefig(filepath)


df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
# print(EDA().get_columns_from_df(df=df))
# print(EDA().get_num_columns_rows_from_df(df=df))
# EDA().missing_values_report(df=df, filepath="reports/eda/missing_values_report_RAW_recipes.jpeg")
# print(EDA().check_for_duplicates(df=df))
# EDA().data_type_report(df=df, filepath="reports/eda/data_types_report_RAW_recipes.jpeg")
df_interactions = FileHandler().csv_to_df(filepath="data/raw/RAW_interactions.csv")
# EDA().rating_distribution(df=df_interactions, filepath="reports/eda/rating_distribution_report_RAW_interactions.jpeg")
# EDA().user_interaction_distribution(df=df_interactions, filepath="reports/eda/user_interaction_distribution_report_RAW_interactions.jpeg")
# EDA().recipe_popularity(df=df_interactions, filepath="reports/eda/recipe_popularity_report_RAW_interactions.jpeg")
EDA().cooking_time_distribution(df=df_recipes, filepath="reports/eda/cooking_time_distribution_report_RAW_recipes.jpeg")
# EDA().ingredients_count_distribution(df=df_recipes, filepath="reports/eda/ingredients_count_RAW_recipes.jpeg")
# EDA().step_count_distribution(df=df_recipes, filepath="reports/eda/step_count_distribution_RAW_recipes.jpeg")