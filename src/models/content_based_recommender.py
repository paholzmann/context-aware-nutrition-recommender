import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data.file_handler import FileHandler
from ..data.preprocess import Preprocessor
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
        features = ["name", "minutes", "tags", "steps", "n_steps", "ingredients", "n_ingredients"]
        id = ["id"]
        columns = id + features
        return df[columns]
    
    def convert_list_to_str(self, df: pd.DataFrame, cols_to_convert: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols_to_convert:
            df[col] = df[col].apply(lambda x: ', '.join(ast.literal_eval(x)))
        return df
    
    def combine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["combined_text"] = df["tags"] + " " +  df["ingredients"] + " " + df["steps"]
        return df
    
    def apply_tfidf(self, df: pd.DataFrame):
        vectorizer = TfidfVectorizer(min_df=5,
            max_df=0.8, max_features=30000,
            stop_words="english", ngram_range=(1,2)
        )
        X = vectorizer.fit_transform(df["combined_text"])
        return X, vectorizer
    
    def get_similar_recipes(self, df, X, recipe_id, top_k):
        idx_list = df.index[df["id"] == recipe_id].to_list()
        idx = idx_list[0]
        query_vector = X[idx]
        similarities = cosine_similarity(query_vector, X).flatten()
        scores = list(enumerate(similarities))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:top_k+1]
        return scores

    def explain_similarity(self, idx1, idx2, X, feature_names, top_k):
        vec1 = X[idx1].toarray()[0]
        vec2 = X[idx2].toarray()[0]
        contributions = vec1 * vec2
        indices = contributions.argsort()[::-1]
        explanation = []
        for i in indices:
            if contributions[i] > 0:
                explanation.append(feature_names[i])
            if len(explanation) == top_k:
                break
        return explanation
    
    def get_prediction(self, recipe_id, df, X, vectorizer, top_k=10):
        results = self.get_similar_recipes(df=df, recipe_id=recipe_id, X=X, top_k=top_k)
        feature_names = vectorizer.get_feature_names_out()
        print(f"Query: {df.iloc[recipe_id]["name"]}")
        for i, score in results:
            explanation = self.explain_similarity(idx1=recipe_id, idx2=i, X=X, feature_names=feature_names, top_k=top_k)
            print(f"{df.iloc[i]["name"]} (Score: {score:.3f})")
            print(f"Similar features: {', '.join(explanation)}")

cbr = ContentBasedRecommender()
df_recipes = FileHandler().csv_to_df(filepath="data/raw/RAW_recipes.csv")
df = cbr.feature_definition(df=df_recipes)
df = cbr.convert_list_to_str(df=df, cols_to_convert=["tags", "steps", "ingredients"])
df= Preprocessor().clean_text(df=df, cols_to_clean=["tags", "steps", "ingredients"])
df = cbr.combine_features(df=df)
X, vectorizer = cbr.apply_tfidf(df=df)
cbr.get_prediction(recipe_id=31490, df=df, X=X, vectorizer=vectorizer)