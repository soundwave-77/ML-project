# %%
# import sys

# sys.path.append("../")

from pathlib import Path
from typing import Optional
import gc

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib
from src.embeddings.loader import (
    embedding_dict_to_df,
    load_embeddings,
    load_json_embedding_as_dict,
    load_text_embeddings_json,
    load_text_embeddings_parquet,
)

pd.set_option("display.max_columns", None)


import numpy as np


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def fill_missing_values(df, cols):
    for col in cols:
        df[col] = df[col].fillna("")
    return df


def preprocess_data_for_model(df, model_name):
    cat_features = df.select_dtypes(include="object").columns
    df = fill_missing_values(df, cat_features)

    # Fill missing values in numerical features
    num_features = df.select_dtypes(include="number").columns
    for col in num_features:
        df[col] = df[col].fillna(df[col].median())

    if model_name in ["Ridge", "LightGBM"]:
        # Count encoding
        for col in cat_features:
            count_map = df[col].value_counts().to_dict()
            df[col] = df[col].map(count_map)
    elif model_name == "CatBoost":
        df["image_top_1"] = df["image_top_1"].astype("str")
    return df


def df_with_nested_embeddings(embed_path: Path, embed_name: str):
    """Add embeddings to dataframe

    embeddings as list in column
    """
    embeddings = load_json_embedding_as_dict(embed_path, embed_name)
    embeddings_df = embedding_dict_to_df(embeddings)
    return embeddings_df


def df_with_text_features_as_separate_columns(df, embed_path: Path, embed_name: str):
    text_embeddings = load_text_embeddings_json(embed_path)
    embeddings_df = df["item_id"].apply(lambda x: pd.Series(text_embeddings[x]))
    embeddings_df = embeddings_df.rename(columns=lambda x: f"title_embedding_{x+1}")
    return embeddings_df


def add_text_statistics(df):
    print("==== Adding text statistics ====")

    # Text length
    df["title_len"] = df["title"].apply(len)
    df["description_len"] = df["description"].apply(len)

    # Word count
    df["title_word_count"] = df["title"].apply(lambda x: len(x.split()))
    df["description_word_count"] = df["description"].apply(lambda x: len(x.split()))

    # Unique words
    df["title_unique_words"] = df["title"].apply(lambda x: len(set(x.split())))
    df["description_unique_words"] = df["description"].apply(
        lambda x: len(set(x.split()))
    )

    # Intersection of unique words
    df["common_unique_words"] = df.apply(
        lambda row: len(set(row["title"].split()) & set(row["description"].split())),
        axis=1,
    )

    # Difference in unique words
    df["title_unique_to_description"] = df.apply(
        lambda row: len(set(row["title"].split()) - set(row["description"].split())),
        axis=1,
    )
    df["description_unique_to_title"] = df.apply(
        lambda row: len(set(row["description"].split()) - set(row["title"].split())),
        axis=1,
    )
    return df


def add_image_statistics(df, file_path):
    print("==== Adding image statistics ====")

    img_stats = pd.read_parquet(file_path)
    df = pd.merge(df, img_stats, on="image", how="left")
    return df


def add_text_features(df, embedding_type, embed_add_as_separate_columns):
    print("==== Adding text features ====")

    embed_features = []
    print("LOADING TEXT EMBEDDINGS")
    print(f"embedding_type: {embedding_type}")
    print(f"embed_add_as_separate_columns: {embed_add_as_separate_columns}")

    if embedding_type == "tfidf":
        # Load TF-IDF embeddings
        title_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/tfidf/train/title_embeddings.parquet"
        ).expanduser()
        description_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/tfidf/train/description_embeddings.parquet"
        ).expanduser()

        # Add title embeddings
        title_embeddings = load_text_embeddings_parquet(title_embed_path)
        title_embeddings.columns = [
            f"title_embedding_{i}" for i in range(title_embeddings.shape[1])
        ]

        # Add description embeddings
        description_embeddings = load_text_embeddings_parquet(description_embed_path)
        description_embeddings.columns = [
            f"description_embedding_{i}" for i in range(description_embeddings.shape[1])
        ]
        df = pd.concat(
            [df.reset_index(drop=True), title_embeddings, description_embeddings],
            axis=1,
        )

        del title_embeddings
        del description_embeddings
        gc.collect()

    elif embedding_type == "fasttext":
        # Load FastText embeddings
        title_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/fasttext/train/title.parquet"
        ).expanduser()
        description_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/fasttext/train/description.parquet"
        ).expanduser()

        # Add title embeddings
        title_embeddings = load_text_embeddings_parquet(title_embed_path)
        title_embeddings.columns = [
            f"title_embedding_{i}" for i in range(title_embeddings.shape[1])
        ]

        # Add description embeddings
        description_embeddings = load_text_embeddings_parquet(description_embed_path)
        description_embeddings.columns = [
            f"description_embedding_{i}" for i in range(description_embeddings.shape[1])
        ]
        df = pd.concat(
            [df.reset_index(drop=True), title_embeddings, description_embeddings],
            axis=1,
        )

        del title_embeddings
        del description_embeddings
        gc.collect()

    elif embedding_type == "rubert":
        # Load RuBERT embeddings
        title_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json"
        ).expanduser()
        description_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/description_embeddings_reduced_train.json"
        ).expanduser()

        # PREVIOUS CODE
        # Add title embeddings
        # title_embeddings = load_text_embeddings_json(title_embed_path)
        # title_df = df["item_id"].apply(lambda x: pd.Series(title_embeddings[x]))
        # title_df = title_df.rename(columns=lambda x: f"title_embedding_{x+1}")
        # df = df.join(title_df, how="left")

        # Add description embeddings
        # description_embeddings = load_text_embeddings_json(description_embed_path)
        # description_df = df["item_id"].apply(lambda x: pd.Series(description_embeddings[x]))
        # description_df = description_df.rename(columns=lambda x: f"description_embedding_{x+1}")
        # df = df.join(description_df, how="left")

        # NEW CODE

        if embed_add_as_separate_columns:  # each embedding feature is a separate column
            embed_df = df_with_text_features_as_separate_columns(
                df, title_embed_path, "title_embedding_rubert"
            )
            df = df.join(embed_df, how="left")
        else:  # all embeddings are in a single column
            embed_features.append("title_embedding_rubert")
            embed_df = df_with_nested_embeddings(
                title_embed_path, "title_embedding_rubert"
            )
            df = df.join(embed_df, on="item_id")

        if embed_add_as_separate_columns:  # each embedding feature is a separate column
            embed_df = df_with_text_features_as_separate_columns(
                df, description_embed_path, "description_embedding_rubert"
            )
            df = df.join(embed_df, how="left")
        else:  # all embeddings are in a single column
            embed_features.append("description_embedding_rubert")
            embed_df = df_with_nested_embeddings(
                description_embed_path, "description_embedding_rubert"
            )
            df = df.join(embed_df, on="item_id")

        del embed_df
        gc.collect()
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return df, embed_features


def add_image_features(
    df, embedding_type: str, use_truncated_embeddings: bool, embed_add_as_separate_columns: bool
):
    print("==== Adding image features ====")
    if embedding_type == "clip":
        embed_path = Path("~/Yandex.Disk/hse_ml_avito/vector_store/clip/train.npz").expanduser()
    elif embedding_type == "resnet":
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/resnet/embeddings_train_merged.npz"
        ).expanduser()
    elif embedding_type == "dinov2":
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/dinov2/train.npz"
        ).expanduser()
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    print(f"embed_path: {embed_path}")
    embeddings = load_embeddings(embed_path)
    print("read embeddings")

    print('going to truncate')
    if use_truncated_embeddings:
        tsvd = TruncatedSVD(32)
        embed = tsvd.fit_transform(embeddings["embeddings"])
        joblib.dump(tsvd, "tsvd_img_embeddings.joblib")

        del tsvd
    else:
        embed = embeddings["embeddings"]
    image_names = embeddings["images"]
    del embeddings
    gc.collect()
    print('truncated embeddings')

    embed_features = []
    if embed_add_as_separate_columns:  # each embedding feature is a separate column
        print('going to add separate columns')
        embed_df = pd.DataFrame(embed, index=image_names)
        embed_cols = [f"image_embedding_{i+1}" for i in range(embed_df.shape[1])]
        embed_df.columns = embed_cols
        df = df.merge(embed_df, left_on="image", right_index=True, how="left")
        df[embed_cols] = df[embed_cols].fillna(0)
        df[embed_cols] = df[embed_cols].astype(np.float16)
    else:
        print('going to add as a single column')
        embed_features.append(f"{embedding_type}_image_embeddings")
        embed_df = pd.DataFrame(list(embed), index=image_names)
        df = df.merge(embed_df, left_on="image", right_index=True, how="left")
        df[embed_cols] = df[embed_cols].fillna(0)
        df[embed_cols] = df[embed_cols].astype(np.float16)
    del embed_df
    gc.collect()
    return df, embed_features


def load_and_preprocess_data(
    model_name: str,
    data_path: Path,
    text_embeddings_type: Optional[str],
    image_embeddings_type: Optional[str],
    use_reduced_rubert_embeddings: bool = False,
    embed_add_as_separate_columns: bool = False,
    use_truncated_embeddings: bool = True,
    use_image_quality_features: bool = False,
):
    print("Parameters")
    print(f"model_name: {model_name}")
    print(f"data_path: {data_path}")
    print(f"text_embeddings_type: {text_embeddings_type}")
    print(f"image_embeddings_type: {image_embeddings_type}")
    print(f"use_reduced_rubert_embeddings: {use_reduced_rubert_embeddings}")
    print(f"embed_add_as_separate_columns: {embed_add_as_separate_columns}")
    print(f"use_truncated_embeddings: {use_truncated_embeddings}")

    # %%
    # data_path = "../data/raw/train.csv"
    # add_text_features = True
    # add_image_features = False
    # use_reduced_rubert_embeddings = False

    # %%
    print("==== Preparing Data ====")
    # data_path = "../data/raw/train.csv"
    df = pd.read_csv(data_path)

    # Fill missing values in categorical features
    cat_features = df.select_dtypes(include="object").columns
    df = fill_missing_values(df, cat_features)

    if text_embeddings_type is not None:
        df = add_text_statistics(df)

    if use_image_quality_features:
        img_stats_path = Path(
            "~/Yandex.Disk/hse_ml_avito/image_statistics.parquet"
        ).expanduser()
        df = add_image_statistics(df, img_stats_path)

    # Cosine similarity
    if text_embeddings_type is not None:
        cos_sim_path = Path(
            "~/Yandex.Disk/hse_ml_avito/title_description_sim_train.csv"
        ).expanduser()
        df_sim = pd.read_csv(cos_sim_path)
        df = pd.merge(df, df_sim, on="item_id", how="left")

    df = preprocess_data_for_model(df, model_name)
    # %%
    embed_features = []

    if image_embeddings_type is not None:
        df, embed_feat = add_image_features(
            df,
            image_embeddings_type,
            use_truncated_embeddings,
            embed_add_as_separate_columns,
        )
        if len(embed_feat) > 0:
            embed_features.append(embed_feat)

    df = reduce_mem_usage(df)

    if text_embeddings_type is not None:
        df, embed_feat = add_text_features(
            df, text_embeddings_type, embed_add_as_separate_columns
        )
        if len(embed_feat) > 0:
            embed_features.append(embed_feat)

    df = df.drop(columns=["title", "description"])
    gc.collect()

    # %%
    # drop unused(for now) columns
    # 'params', 'price_log', 'deal_prob_cat',
    drop_cols = [
        "image",
        "item_id",
        "user_id",
        "activation_date",
    ]
    df.drop(drop_cols, axis=1, inplace=True)

    df = reduce_mem_usage(df)

    X = df.drop(columns=["deal_probability"])
    y = df["deal_probability"]

    # X = X.apply(pd.to_numeric, errors='coerce', downcast='float')
    # y = y.astype(np.float16)

    print("==== Data preprocessed successfully! ====")

    cat_features = [
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        "user_type",
        "image_top_1",
    ]
    return {
        "X": X,
        "y": y,
        "cat_features": cat_features,
        "embed_features": embed_features,
    }


def preprocess_data_ridge(data_path: Path):
    print("==== Preparing Data ====")
    df_train = pd.read_csv(data_path)

    df_train.drop(
        ["image", "item_id", "user_id", "activation_date", "title", "description"],
        axis=1,
        inplace=True,
    )

    # Count encoding
    cat_features = df_train.select_dtypes(include="object").columns
    for col in cat_features:
        df_train[col] = df_train[col].fillna("")

    for col in cat_features:
        count_map = df_train[col].value_counts().to_dict()
        df_train[col] = df_train[col].map(count_map)

    num_features = df_train.select_dtypes(include="number").columns
    for col in num_features:
        df_train[col] = df_train[col].fillna(df_train[col].median())

    X = df_train.drop(columns=["deal_probability"])
    y = df_train["deal_probability"]

    print("==== Data preprocessed successfully! ====")

    return {"X": X, "y": y, "cat_features": cat_features}


# if __name__ == "__main__":
#     data = load_and_preprocess_data("data/raw/train.csv")
#     print(data["X"].head(3))
#     print(data["y"].head(3))
#     print(data["cat_features"])
