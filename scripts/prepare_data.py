# %%
from pathlib import Path

import pandas as pd

from src.embeddings.loader import load_text_embeddings_json, load_embeddings

pd.set_option("display.max_columns", None)


def fill_missing_values(df, cols):
    for col in cols:
        df[col] = df[col].fillna("")
    return df


def load_and_preprocess_data(
    data_path: Path, add_text_features: bool = False, add_image_features: bool = False
):
    # %%
    print("==== Preparing Data ====")
    # data_path = "../data/raw/train.csv"
    df = pd.read_csv(data_path)
    
    # Fill missing values in categorical features
    cat_features = df.select_dtypes(include="object").columns
    df = fill_missing_values(df, cat_features)

    # Count encoding
    for col in cat_features:
        count_map = df[col].value_counts().to_dict()
        df[col] = df[col].map(count_map)

    # Fill missing values in numerical features
    num_features = df.select_dtypes(include="number").columns
    for col in num_features:
        df[col] = df[col].fillna(df[col].median())


    # %%
    if add_text_features:
        print("==== Adding text features ====")
        # add title embeddings
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json"
        ).expanduser()

        # add text features
        text_embeddings = load_text_embeddings_json(embed_path)
        embeddings_df = df["item_id"].apply(lambda x: pd.Series(text_embeddings[x]))
        embeddings_df = embeddings_df.rename(columns=lambda x: f"title_embedding_{x+1}")
        df = df.join(embeddings_df, how="left")

        # add description embeddings
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/description_embeddings_reduced_train.json"
        ).expanduser()
        text_embeddings = load_text_embeddings_json(embed_path)
        embeddings_df = df["item_id"].apply(lambda x: pd.Series(text_embeddings[x]))
        embeddings_df = embeddings_df.rename(
            columns=lambda x: f"description_embedding_{x+1}"
        )
        df = df.join(embeddings_df, how="left")

    if add_image_features:
        print("==== Adding image features ====")
        # add image embeddings
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/resnet/embeddings_train_merged.npz"
        ).expanduser()
        embeddings = load_embeddings(embed_path)
        embeddings_df = df["item_id"].apply(lambda x: pd.Series(embeddings[x]))
        embeddings_df = embeddings_df.rename(columns=lambda x: f"image_embedding_{x+1}")
        df = df.join(embeddings_df, how="left")

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
    data = load_and_preprocess_data("data/raw/train.csv")
    print(data["X"].head(3))
    print(data["y"].head(3))
