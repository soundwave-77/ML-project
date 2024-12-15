# %%
from pathlib import Path

import pandas as pd

from embedding_utils import load_text_embeddings_json

pd.set_option("display.max_columns", None)


def prepare_price_col(df):
    """
    Function to fill missing values in price column with mean price
    """
    df["price"] = df["price"].fillna(df["price"].mean())
    return df


def fill_missing_values(df, cols):
    for col in cols:
        df[col] = df[col].fillna("")
    return df


def load_and_preprocess_data(data_path: Path, add_text_features: bool = False):
    # %%
    print("==== Preparing Data ====")
    # data_path = "../data/raw/train.csv"
    df = pd.read_csv(data_path)
    # prepare price column
    df = prepare_price_col(df)
    # fill missing values
    col_list = ["param_1", "param_2", "param_3", "image_top_1", "title", "description"]
    df = fill_missing_values(df, col_list)
    # convert image_top_1 to string
    df["image_top_1"] = df["image_top_1"].astype("str")

    # %%
    if add_text_features:
        print("==== Adding text features ====")
        embed_path = Path(
            "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json"
        ).expanduser()

        # add text features
        text_embeddings = load_text_embeddings_json(embed_path)
        embeddings_df = df["item_id"].apply(lambda x: pd.Series(text_embeddings[x]))
        embeddings_df = embeddings_df.rename(columns=lambda x: f"embedding_{x+1}")
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
    return X, y, cat_features


if __name__ == "__main__":
    X, y, cat_features = load_and_preprocess_data("data/raw/train.csv")
    print(X.head(3))
    print(y.head(3))
    print(cat_features)
