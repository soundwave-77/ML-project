import pandas as pd
import numpy as np
import argparse
import joblib
from nltk.corpus import stopwords
from catboost import CatBoostRegressor, Pool


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def preprocess_data(df, stop_words, cat_features):
    # Fill missing values
    df["price"] = df["price"].fillna(df["price"].mean())
    for col in ["param_1", "param_2", "param_3", "image_top_1", "title", "description"]:
        df[col] = df[col].fillna("")

    # Convert categorical columns to string
    df["image_top_1"] = df["image_top_1"].astype("str")

    # Drop unnecessary columns
    df = df.drop(
        ["image", "item_id", "user_id", "activation_date", "title", "description"],
        axis=1,
        errors="ignore",
    )

    return df


def main(model_path, input_csv, output_csv):
    stopWords = set(stopwords.words("russian"))
    # Load the trained model
    model = CatBoostRegressor()
    model.load_model(model_path)

    # Load the input data
    df = pd.read_csv(input_csv)

    # Define categorical features
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

    # Preprocess the data
    X = preprocess_data(df, stopWords, cat_features)

    # Create Pool for CatBoost
    pool = Pool(X, cat_features=cat_features)

    # Make predictions
    predictions = model.predict(pool)

    # Clip predictions to [0, 1]
    deal_probability = np.clip(predictions, 0, 1)

    # Save predictions to output CSV
    df["deal_probability"] = deal_probability
    df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained CatBoost model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained CatBoost model file.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the input CSV file for prediction.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path to save the output predictions.",
    )

    args = parser.parse_args()
    main(args.model_path, args.input_csv, args.output_csv)
