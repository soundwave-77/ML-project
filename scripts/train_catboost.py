import argparse

import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from clearml import Task
from tqdm import tqdm
from prepare_data import load_and_preprocess_data
from utils import save_feature_importance, rmse


def train_model(model_name, task_name, X, y, cat_features):
    # Initialize ClearML task
    task = Task.init(project_name="avito_sales_prediction", task_name=task_name)

    task.connect(
        {
            "dataset_size": X.shape,
            "numerical_features": X.select_dtypes(include="number").columns.to_list(),
            "categorical_features": cat_features,
        }
    )

    # Cross-validation
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    stratify_labels = (y.round(2) * 100).astype(int)
    FOLD_LIST = list(skf.split(stratify_labels, stratify_labels))

    for train_idx, val_idx in tqdm(FOLD_LIST):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        eval_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(metric_period=50, early_stopping_rounds=5)
        model.fit(train_pool, eval_set=eval_pool)

        preds = model.predict(X_val)
        score = rmse(y_val, preds)
        scores.append(score)

    mean_score = np.mean(scores)

    print(f"Mean RMSE across folds: {mean_score}")
    task.connect({"mean_rmse_across_folds": mean_score})

    # Train model with best hyperparameters
    model = CatBoostRegressor(
        metric_period=50, early_stopping_rounds=5, iterations=1000
    )
    model.fit(X, y, cat_features=cat_features)

    params = model.get_params()
    # Log hyperparameters to ClearML
    task.connect(params)

    # Evaluate final model
    preds = model.predict(X)
    rmse_score = rmse(preds, y)
    task.connect({"train_rmse": rmse_score})

    save_feature_importance(model, X, task, task_name)
    return model, rmse_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on provided dataset")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training dataset CSV file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CatBoost",
        help="Name of the model to be trained",
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the ClearML task"
    )

    args = parser.parse_args()

    X, y, cat_features = load_and_preprocess_data(args.train_path, add_text_features=True)
    train_model(args.model_name, args.task_name, X, y, cat_features)
