import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from ..data.file_handler import FileHandler
from ..features.recipe_features import RecipeFeatures

class ContentBasedRecommender:
    def __init__(self):
        """
        TF-IDF
        Cosine Similarity
        Top-k Ranking
        """
        pass

    def feature_definition(self, df: pd.DataFrame) -> pd.DataFrame:
        """ works with recipe df """
        features = ["minutes", "tags", "steps", "n_steps", "ingredients", "n_ingredients"]
        id = ["id"]
        columns = id + features
        return df[columns]
    
    def convert_list_to_str(self, df: pd.DataFrame, cols_to_convert: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols_to_convert:
            df[col] = df[col].apply(lambda x: ', '.join(ast.literal_eval(x)))
        return df

    def apply_tfidf(self, df: pd.DataFrame) -> None:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), max_df=2)
        tfidf_vector = tfidf_vectorizer.fit_transform(df)
        print(tfidf_vector)

df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
df = ContentBasedRecommender().feature_definition(df=df_recipes)
df = ContentBasedRecommender().convert_list_to_str(df=df, cols_to_convert=["tags", "steps", "ingredients"])
# print(df)
ContentBasedRecommender().apply_tfidf(df=df)