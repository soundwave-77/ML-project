# %%
from pathlib import Path

import pandas as pd

from src.embeddings.loader import load_text_embeddings_json, load_text_embeddings_parquet, load_embeddings

pd.set_option("display.max_columns", None)


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


def add_text_statistics(df):
    print("==== Adding text statistics ====")

    # Text length
    df['title_len'] = df['title'].apply(len)
    df['description_len'] = df['description'].apply(len)

    # Word count
    df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
    df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))

    # Unique words
    df['title_unique_words'] = df['title'].apply(lambda x: len(set(x.split())))
    df['description_unique_words'] = df['description'].apply(lambda x: len(set(x.split())))

    # Intersection of unique words
    df['common_unique_words'] = df.apply(
        lambda row: len(set(row['title'].split()) & set(row['description'].split())), axis=1
    )

    # Difference in unique words
    df['title_unique_to_description'] = df.apply(
        lambda row: len(set(row['title'].split()) - set(row['description'].split())), axis=1
    )
    df['description_unique_to_title'] = df.apply(
        lambda row: len(set(row['description'].split()) - set(row['title'].split())), axis=1
    )
    return df


def add_image_statistics(df, file_path):
    print("==== Adding image statistics ====")

    img_stats = pd.read_parquet(file_path)
    df = pd.merge(df, img_stats, on='image', how='left')
    return df


def add_text_features(df, embedding_type):
    print("==== Adding text features ====")

    if embedding_type == "tfidf":
        # Load TF-IDF embeddings
        title_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/tfidf/train/title_embeddings.parquet"
        ).expanduser()
        description_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/fasttext/train/description_embeddings.parquet"
        ).expanduser()

        # Add title embeddings
        title_embeddings = load_text_embeddings_parquet(title_embed_path)
        title_embeddings.columns = [f"title_embedding_{i}" for i in range(title_embeddings.shape[1])]

        # Add description embeddings
        description_embeddings = load_text_embeddings_parquet(description_embed_path)
        description_embeddings.columns = [f"description_embedding_{i}" for i in range(description_embeddings.shape[1])]
        df = pd.concat([df.reset_index(drop=True), title_embeddings, description_embeddings], axis=1)
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
        title_embeddings.columns = [f"title_embedding_{i}" for i in range(title_embeddings.shape[1])]

        # Add description embeddings
        description_embeddings = load_text_embeddings_parquet(description_embed_path)
        description_embeddings.columns = [f"description_embedding_{i}" for i in range(description_embeddings.shape[1])]
        df = pd.concat([df.reset_index(drop=True), title_embeddings, description_embeddings], axis=1)
    elif embedding_type == "rubert":
        # Load RuBERT embeddings
        title_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json"
        ).expanduser()
        description_embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/description_embeddings_reduced_train.json"
        ).expanduser()

        # Add title embeddings
        title_embeddings = load_text_embeddings_json(title_embed_path)
        title_df = df["item_id"].apply(lambda x: pd.Series(title_embeddings[x]))
        title_df = title_df.rename(columns=lambda x: f"title_embedding_{x+1}")
        df = df.join(title_df, how="left")

        # Add description embeddings
        description_embeddings = load_text_embeddings_json(description_embed_path)
        description_df = df["item_id"].apply(lambda x: pd.Series(description_embeddings[x]))
        description_df = description_df.rename(columns=lambda x: f"description_embedding_{x+1}")
        df = df.join(description_df, how="left")
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return df


def add_image_features(df, embedding_type):
    print("==== Adding image features ====")

    if embedding_type == 'resnet':
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/resnet/embeddings_train_merged.npz"
        ).expanduser()
        embeddings = load_embeddings(embed_path)
        embeddings_df = df["item_id"].apply(lambda x: pd.Series(embeddings[x]))
        embeddings_df = embeddings_df.rename(columns=lambda x: f"image_embedding_{x+1}")
        df = df.join(embeddings_df, how="left")
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return df


def load_and_preprocess_data(
    model_name: str,
    data_path: Path,
    text_embeddings_type: None,
    image_embeddings_type: None
):
    # %%
    print("==== Preparing Data ====")
    # data_path = "../data/raw/train.csv"
    df = pd.read_csv(data_path)

    # Fill missing values in categorical features
    cat_features = df.select_dtypes(include="object").columns
    df = fill_missing_values(df, cat_features)
    
    df = add_text_statistics(df)
    img_stats_path = Path(
        "~/Yandex.Disk/hse_ml_avito/image_statistics.parquet"
    ).expanduser()
    df = add_image_statistics(df, img_stats_path)

    # Cosine similarity
    cos_sim_path = Path(
        "~/Yandex.Disk/hse_ml_avito/title_description_sim_train.csv"
    ).expanduser()
    df_sim = pd.read_csv(cos_sim_path)
    df = pd.merge(df, df_sim, on='item_id', how='left')

    df = preprocess_data_for_model(df, model_name)
    # %%
    if text_embeddings_type is not None:
        df = add_text_features(df, text_embeddings_type)

    if image_embeddings_type is not None:
        df = add_image_features(df, image_embeddings_type)

    # %%
    # drop unused(for now) columns
    # 'params', 'price_log', 'deal_prob_cat',
    drop_cols = [
        "image",
        "item_id",
        "user_id",
        "activation_date",
        "title",
        "description",
    ]
    df.drop(drop_cols, axis=1, inplace=True)

    X = df.drop(columns=["deal_probability"])
    y = df["deal_probability"]

    print("==== Data preprocessed successfully! ====")

    return {"X": X, "y": y}


if __name__ == "__main__":
    data = load_and_preprocess_data("CatBoost", "data/raw/train.csv")
    print(data["X"].head(3))
    print(data["y"].head(3))
