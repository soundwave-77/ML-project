import numpy as np
import pandas as pd
import json
import pickle
import h5py


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


def load_text_embeddings_h5(filepath: str, column_name: str) -> pd.DataFrame:
    h5 = h5py.File(filepath, "r")
    df = pd.DataFrame(np.array(h5["embeddings"]), index=h5["item_ids"])
    df.index = df.index.astype("str")
    df = df.rename(columns=lambda x: f"{column_name}_embedding_{x+1}")
    return df


def load_h5_embedding_as_dict(filepath: str, column_name: str):
    h5 = h5py.File(filepath, "r")
    return {column_name: list(h5["embeddings"]), "item_ids": h5["item_ids"]}


def load_json_embedding_as_dict(filepath: str, column_name: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    return {column_name: list(data.values()), "item_ids": data.keys()}


def embedding_dict_to_df(embeddings: dict):
    dff = pd.DataFrame.from_dict(embeddings)
    dff = dff.set_index('item_ids')
    dff.index = dff.index.astype("str")
    return dff


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


if __name__ == "__main__":
    # embeddings = load_h5_embedding_as_dict(
    #     "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_train.h5",
    #     "title_rubert_embeddings"
    # )
    embeddings = load_json_embedding_as_dict(
        "/home/qb/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json",
        "title_rubert_embeddings_short"
    )

    df = embedding_dict_to_df(embeddings)
    print(df.head(2))
