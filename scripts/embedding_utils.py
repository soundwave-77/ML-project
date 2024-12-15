import numpy as np
import pandas as pd
import json
import pickle


def load_embeddings(filepath: str):
    """
        Returns an array with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/resnet/embeddings_train_merged.npz
    """
    embed = np.load(filepath)
    return embed  # keys: ['embeddings', 'images'] 


def load_text_embeddings_json(filepath: str):
    """
        Returns a dictionary with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def load_text_embeddings_pickle(filepath: str):
    """
        Returns a dictionary with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.pkl
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_text_embeddings_parquet(filepath: str):
    """
        Returns a dataframe with index as item_id and column for embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.parquet
    """
    df = pd.read_parquet(filepath)
    return df
