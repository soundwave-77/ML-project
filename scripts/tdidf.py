import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import joblib


class TfidfSVDModel:
    def __init__(self, tfidf_params, svd_params=None):
        self.tfidf = TfidfVectorizer(**tfidf_params)
        self.svd = TruncatedSVD(**svd_params) if svd_params else None

    def fit(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        if self.svd:
            self.svd.fit(tfidf_matrix)
        return self

    def transform(self, texts):
        tfidf_matrix = self.tfidf.transform(texts)
        if self.svd:
            return self.svd.transform(tfidf_matrix)
        return tfidf_matrix

    def fit_transform(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        if self.svd:
            return self.svd.fit_transform(tfidf_matrix)
        return tfidf_matrix


def process_and_save_tfidf(
        df,
        text_columns,
        max_features,
        ngram_range=(1, 2),
        output_dir="vec/tfidf",
        n_components=None
):
    """
    Processes text data using a TfidfVectorizer and optional TruncatedSVD,
    then saves the combined model to a .pkl file and embeddings to Parquet.
    """
    df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

    tfidf_params = {
        "sublinear_tf": True,
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
        "stop_words": stopwords.words("russian"),
        "max_features": max_features,
        "ngram_range": ngram_range,
    }

    svd_params = {"n_components": n_components, "random_state": 42} if n_components else None

    model = TfidfSVDModel(tfidf_params, svd_params)

    X_texts = model.fit_transform(df["combined_text"])

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"tfidf_svd_model.pkl")
    joblib.dump(model, model_path)

    columns_str = "_".join(text_columns)
    embeddings_path = os.path.join(output_dir, f"{columns_str}_embeddings.parquet")

    embeddings_df = pd.DataFrame(X_texts, index=df.index)
    embeddings_df.to_parquet(embeddings_path, index=True)

    print(f"Combined TF-IDF + SVD model saved to: {model_path}")
    print(f"Embeddings saved to: {embeddings_path}")
