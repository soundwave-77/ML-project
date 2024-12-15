import argparse
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from clearml import Task
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# local modules
from prepare_data import load_and_preprocess_data, preprocess_data_ridge
from src.utils import is_gpu_available, rmse, save_feature_importance

Task.set_offline(True)


def train_catboost(cat_features, X_train, y_train, X_val=None, y_val=None):
    task_type = "GPU" if is_gpu_available() else "CPU"
    print(f"Using {task_type} for training")

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    if X_val is not None and y_val is not None:
        eval_pool = Pool(X_val, y_val, cat_features=cat_features)
    else:
        eval_pool = None
    model = CatBoostRegressor(
        metric_period=50,
        early_stopping_rounds=5,
        iterations=1000,
        task_type=task_type,
    )
    model.fit(train_pool, eval_set=eval_pool)
    return model


def train_ridge(X_train, y_train, X_val, y_val):
    model = make_pipeline(StandardScaler(), Ridge())
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, **params):
    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=5, verbose=True), log_evaluation(50)],
    )
    return model


def train_model(model_name, task_name, X, y, cat_features):
    # Initialize ClearML task
    task = Task.init(project_name="avito_sales_prediction", task_name=task_name)

    task.connect(
        {
            "model_name": model_name,
            "dataset_size": X.shape,
            "numerical_features": X.select_dtypes(include="number").columns.to_list(),
            "categorical_features": X.select_dtypes(include="object").columns.to_list(),
        }
    )
    logger = task.get_logger()

    # Cross-validation
    scores = []
    skf = KFold(n_splits=3, shuffle=True, random_state=42)
    FOLD_LIST = list(skf.split(X, y))

    for train_idx, val_idx in tqdm(FOLD_LIST):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]

        if model_name == "CatBoost":
            model = train_catboost(cat_features, X_train, y_train, X_val, y_val)
        elif model_name == "Ridge":
            model = train_ridge(X_train, y_train, X_val, y_val)
        elif model_name == "LightGBM":
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 1000,
                "boosting_type": "gbdt",
                "verbose": -1,
            }
            model = train_lightgbm(X_train, y_train, X_val, y_val)

        preds = model.predict(X_val)
        score = rmse(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        scores.append({"rmse": score, "mae": mae, "r2": r2})

    scores = {
        "rmse": np.mean([score["rmse"] for score in scores]),
        "mae": np.mean([score["mae"] for score in scores]),
        "r2": np.mean([score["r2"] for score in scores]),
    }
    for key, value in scores.items():
        logger.report_single_value(name=key, value=value)

    # Evaluate final model
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train model with best hyperparameters
    if model_name == "CatBoost":
        model = train_catboost(
            cat_features, X_train_final, y_train_final, X_val_final, y_val_final
        )
    elif model_name == "Ridge":
        model = train_ridge(X_train_final, y_train_final, X_val_final, y_val_final)

    preds = model.predict(X_val_final)
    rmse_score = rmse(y_val_final, preds)
    mae = mean_absolute_error(y_val_final, preds)
    r2 = r2_score(y_val_final, preds)

    logger.report_single_value(name="RMSE", value=rmse_score)
    logger.report_single_value(name="MAE", value=mae)
    logger.report_single_value(name="R2 Score", value=r2)

    # Log hyperparameters to ClearML
    if model_name == "CatBoost":
        params = model.get_all_params()
    elif model_name == "Ridge":
        params = model.get_params()
    task.connect(params)

    # Log feature importances to ClearML
    if model_name == "CatBoost":
        feature_importances = model.get_feature_importance()
    elif model_name == "Ridge":
        feature_importances = np.abs(model.named_steps["ridge"].coef_)
    elif model_name == "LightGBM":
        feature_importances = model.feature_importances_

    save_feature_importance(feature_importances, X, task, task_name, model_name)

    # Save final model checkpoint
    ckpt_root = Path(f"outputs/models/{model_name}/{task_name}/")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if model_name == "CatBoost":
        final_checkpoint_path = ckpt_root / "final_model.cbm"
        model.save_model(final_checkpoint_path)
    elif model_name == "Ridge":
        final_checkpoint_path = ckpt_root / "final_model.joblib"
        joblib.dump(model, final_checkpoint_path)
    elif model_name == "LightGBM":
        final_checkpoint_path = ckpt_root / "final_model.lgb"
        model.save_model(final_checkpoint_path)
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

    if args.model_name in ["Ridge", "LightGBM"]:
        X, y, cat_features = preprocess_data_ridge(
            args.train_path, add_text_features=False
        )
    elif args.model_name == "CatBoost":
        X, y, cat_features = load_and_preprocess_data(
            args.train_path,
            add_text_features=True,
            # add_image_features=True,
        )
    train_model(args.model_name, args.task_name, X, y, cat_features)
