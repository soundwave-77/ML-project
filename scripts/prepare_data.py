# %%
import pandas as pd

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


def load_and_preprocess_data(data_path):
    # %%

    print("==== Preparing Data ====")

    data_path = "data/raw/train.csv"
    df = pd.read_csv(data_path)
    # prepare price column
    df = prepare_price_col(df)
    # fill missing values
    col_list = ["param_1", "param_2", "param_3", "image_top_1", "title", "description"]
    df = fill_missing_values(df, col_list)
    # convert image_top_1 to string
    df["image_top_1"] = df["image_top_1"].astype("str")

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
    print(X.head())
    print(y.head())
    return X, y, cat_features
