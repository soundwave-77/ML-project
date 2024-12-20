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
            print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        return self

    def transform(self, texts):
        tfidf_matrix = self.tfidf.transform(texts)
        if self.svd:
            return self.svd.transform(tfidf_matrix)
        return tfidf_matrix

    def fit_transform(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        if self.svd:
            self.svd.fit(tfidf_matrix)
            print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
            return self.svd.transform(tfidf_matrix)
        return tfidf_matrix



def process_and_save_tfidf(
        train_df,
        test_df,
        text_columns,
        max_features,
        ngram_range=(1, 2),
        output_dir="vec/tfidf",
        n_components=None
):
    """
    Processes text data using a TfidfVectorizer and optional TruncatedSVD,
    then saves the combined model to a .pkl file and embeddings for both train and test data to Parquet.
    """
    train_df["combined_text"] = train_df[text_columns].fillna("").agg(" ".join, axis=1)
    test_df["combined_text"] = test_df[text_columns].fillna("").agg(" ".join, axis=1)

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

    X_train = model.fit_transform(train_df["combined_text"])
    print(f"Train result shape: {X_train.shape}")
    X_test = model.transform(test_df["combined_text"])
    print(f"Test result shape: {X_test.shape}")

    os.makedirs(output_dir, exist_ok=True)
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    columns_str = "_".join(text_columns)

    model_path = os.path.join(output_dir, f"{columns_str}_tfidf_svd_model.pkl")
    joblib.dump(model, model_path)

    train_embeddings_path = os.path.join(train_output_dir, f"{columns_str}_embeddings.parquet")
    test_embeddings_path = os.path.join(test_output_dir, f"{columns_str}_embeddings.parquet")

    pd.DataFrame(X_train, index=train_df.index).to_parquet(train_embeddings_path, index=True)
    pd.DataFrame(X_test, index=test_df.index).to_parquet(test_embeddings_path, index=True)

    print(f"Combined TF-IDF + SVD model saved to: {model_path}")
    print(f"Train embeddings saved to: {train_embeddings_path}")
    print(f"Test embeddings saved to: {test_embeddings_path}")
