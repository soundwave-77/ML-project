# %%
import sys
sys.path.append("../")

from pathlib import Path

import pandas as pd

from src.embeddings.loader import load_text_embeddings_json, load_embeddings, embedding_dict_to_df, load_json_embedding_as_dict

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


def load_and_preprocess_data(
    model_name: str,
    data_path: Path,
    add_text_features: bool = False,
    add_image_features: bool = False,
    use_reduced_rubert_embeddings: bool = False,
):
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

    df = preprocess_data_for_model(df, model_name)
    # %%
    if add_text_features:
        print("==== Adding text features ====")

        if use_reduced_rubert_embeddings:
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
        else:
            # first way to store embeddings: as separate columns
            # text_embeddings_df = load_text_embeddings_h5(embed_path, "title")
            # df = df.join(text_embeddings_df, how="left")

            # add title embeddings
            # embed_path = Path(
            #     "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_train.h5"
            # ).expanduser()
            embed_path = Path(
                "~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json"
            ).expanduser()

            # second way to store embeddings: as list in column
            # embeddings = load_embeddings_h5(embed_path, 'title_rubert_embeddings')
            embed_name = 'title_rubert_embeddings_short_json'
            embeddings = load_json_embedding_as_dict(embed_path, embed_name)
            embeddings_df = embedding_dict_to_df(embeddings)
            df = df.join(embeddings_df, on='item_id')
            embed_features = [embed_name]

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
    return {"X": X, "y": y, "cat_features": cat_features, "embed_features": embed_features}


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
