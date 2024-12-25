import os

if os.getenv("DEBUG", "0") == "1":
	import debugpy
	debugpy.listen(5678)
	print("Waiting for debugger to attach...")
	debugpy.wait_for_client()
	print("Debugger Connected")

import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import tyro
import optuna
from catboost import CatBoostRegressor, Pool
from clearml import Task
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# local modules
from prepare_data import fill_missing_values, load_and_preprocess_data, preprocess_data_ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import is_gpu_available, rmse, save_feature_importance

if os.getenv("ENABLE_CLEARML", "0") == "0":
    # to disable clearml for debugging
    Task.set_offline(True)


def get_default_hyperparameters(model_name):
    if model_name == "CatBoost":
        # return {
        #     "metric_period": 50,
        #     "early_stopping_rounds": 10,
        #     "iterations": 1000,
        #     "task_type": "GPU" if is_gpu_available() else "CPU",
        # }
        return {
            "metric_period": 50,
            "l2_leaf_reg": 5.332432327271731,
            "learning_rate": 0.10455428370031629,
            "iterations": 1773,
            "depth": 8,
            "task_type": "GPU" if is_gpu_available() else "CPU",
        }

    elif model_name == "LightGBM":
        return {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 1000,
            "boosting_type": "gbdt",
            "verbose": -1,
        }
    elif model_name == "Ridge":
        return {
            "alpha": 1.0
        }
    

def get_hyperparameter_space(model_name, trial):
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "metric_period": 50,
            "early_stopping_rounds": 10,
            "task_type": "GPU" if is_gpu_available() else "CPU",
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1
        }
    elif model_name == "Ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)
        }


def train_catboost(
    X_train, y_train, X_val=None, y_val=None, cat_features=None, embed_features=None, params=None
):
    params = get_default_hyperparameters("CatBoost") if params is None else params
    print(f"Using {params['task_type']} for training")

    train_pool = Pool(
        X_train, y_train, cat_features=cat_features, embedding_features=embed_features
    )
    if X_val is not None and y_val is not None:
        eval_pool = Pool(
            X_val, y_val, cat_features=cat_features, embedding_features=embed_features
        )
    else:
        eval_pool = None
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=eval_pool)
    return model


def train_ridge(X_train, y_train, params=None):
    params = get_default_hyperparameters("Ridge") if params is None else params
    model = make_pipeline(StandardScaler(), Ridge(**params))
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, params=None):
    params = get_default_hyperparameters("LightGBM") if params is None else params

    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        callbacks = [
            early_stopping(stopping_rounds=10, verbose=True),
            log_evaluation(50),
        ]
    else:
        eval_set = None
        callbacks = None

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train, eval_set=eval_set, eval_metric="rmse", callbacks=callbacks
    )
    return model


def optimize_hyperparameters(
    model_name,
    X,
    y,
    FOLD_LIST,
    cat_features,
    embed_features
):
    print("--- Optimizing hyperparameters ---")
    best_scores = []
    best_model = None

    def objective(trial):
        params = get_hyperparameter_space(model_name, trial)
        
        scores = []
        for fold_id, (train_idx, val_idx) in enumerate(FOLD_LIST):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]

            if model_name == "CatBoost":
                model = train_catboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    cat_features=cat_features,
                    embed_features=embed_features,
                    params=params
                )
            elif model_name == "Ridge":
                model = train_ridge(X_train, y_train, params)
            elif model_name == "LightGBM":
                model = train_lightgbm(X_train, y_train, X_val, y_val, params)

            # Ensure y_val is a Pandas Series
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.squeeze()
            elif isinstance(y_val, np.ndarray) and y_val.ndim > 1:
                y_val = y_val.ravel()

            preds = model.predict(X_val)
            score = rmse(y_val, preds)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            scores.append({"rmse": score, "mae": mae, "r2": r2})

        mean_rmse = np.mean([score["rmse"] for score in scores])
        nonlocal best_scores, best_model
        if not best_scores or mean_rmse < np.mean([score["rmse"] for score in best_scores]):
            best_scores = scores
            best_model = model

        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    print(f"Best hyperparameters: {study.best_params}")
    return best_model, best_scores


def train_model(
    model_name,
    task_name,
    X,
    y,
    embed_features=None,
    use_stratified_kfold=False,
    use_hyperparameter_optimization=False
):
    # Initialize ClearML task
    task = Task.init(project_name="avito_sales_prediction", task_name=task_name)

    all_num_features = list(X.select_dtypes(include="number").columns)

    if embed_features is None:
        num_features = all_num_features
    else:
        num_features = set(all_num_features) - set(embed_features)

    cat_features = list(X.select_dtypes(include="object").columns)

    task.connect(
        {
            "model_name": model_name,
            "dataset_size": X.shape,
            "numerical_features": num_features,
            "categorical_features": cat_features,
        }
    )
    logger = task.get_logger()

    # Cross-validation
    folds_num = 3
    if use_stratified_kfold:
        splitter = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=42)

        _y = (y.round(2) * 100).astype(int)
        FOLD_LIST = list(splitter.split(_y, _y))
    else:
        splitter = KFold(n_splits=folds_num, shuffle=True, random_state=42)
        FOLD_LIST = list(splitter.split(X, y))

    # Optuna hyperparameter optimization
    if use_hyperparameter_optimization:
        model, scores = optimize_hyperparameters(model_name, X, y, FOLD_LIST, cat_features, embed_features)
    else:
        scores = []
        for fold_id, (train_idx, val_idx) in enumerate(FOLD_LIST):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]

            if model_name == "CatBoost":
                model = train_catboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    cat_features=cat_features,
                    embed_features=embed_features
                )
            elif model_name == "Ridge":
                model = train_ridge(X_train, y_train)
            elif model_name == "LightGBM":
                model = train_lightgbm(X_train, y_train, X_val, y_val)

            # Ensure y_val is a Pandas Series
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.squeeze()
            elif isinstance(y_val, np.ndarray) and y_val.ndim > 1:
                y_val = y_val.ravel()

            preds = model.predict(X_val)
            score = rmse(y_val, preds)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            scores.append({"rmse": score, "mae": mae, "r2": r2})

        # Log feature importances to ClearML
        if model_name == "CatBoost":
            feature_importances = model.get_feature_importance()
        elif model_name == "Ridge":
            feature_importances = np.abs(model.named_steps["ridge"].coef_)
        elif model_name == "LightGBM":
            feature_importances = model.feature_importances_

        save_feature_importance(
            feature_importances,
            X,
            task,
            task_name,
            model_name,
            f"feature_importance.csv",
        )

    scores = {
        "cv_rmse": np.mean([score["rmse"] for score in scores]),
        "cv_mae": np.mean([score["mae"] for score in scores]),
        "cv_r2": np.mean([score["r2"] for score in scores]),
    }
    for key, value in scores.items():
        logger.report_single_value(name=key, value=value)

    # Log hyperparameters to ClearML
    if model_name == "CatBoost":
        params = model.get_all_params()
    elif model_name in ["Ridge", "LightGBM"]:
        params = model.get_params()
    task.connect(params)

    # Save final model checkpoint
    ckpt_root = Path(f"outputs/models/{model_name}/{task_name}/")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if model_name == "CatBoost":
        final_checkpoint_path = ckpt_root / "final_model.cbm"
        model.save_model(final_checkpoint_path)
    elif model_name in ["Ridge", "LightGBM"]:
        final_checkpoint_path = ckpt_root / "final_model.joblib"
        joblib.dump(model, final_checkpoint_path)


def main(
    train_path: str,
    task_name: str,
    model_name: str,
    use_stratified_kfold: bool,
    use_hyperparameter_optimization: bool,
    embed_add_as_separate_columns: bool,
    use_truncated_embeddings: bool,
    use_prep_data_cache: bool,
    text_embeddings_type: Optional[str],
    image_embeddings_type: Optional[str],
    use_image_quality_features: bool,
):
    task_name = (
        task_name
        + f"_{model_name}"
        + f"_use_stratified_kfold={use_stratified_kfold}"
        + f"_embed_add_as_separate_columns={embed_add_as_separate_columns}"
        + f"_text_embeddings_type={text_embeddings_type}"
        + f"_image_embeddings_type={image_embeddings_type}"
        + f"_use_truncated_embeddings={use_truncated_embeddings}"
    )

    # check if data is already prepared
    x_path = Path(f"outputs/data/{task_name}/X.csv")
    y_path = Path(f"outputs/data/{task_name}/y.csv")
    cat_features_path = Path(f"outputs/data/{task_name}/cat_features.pkl")
    embed_features_path = Path(f"outputs/data/{task_name}/embed_features.pkl")

    x_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        use_prep_data_cache
        and x_path.exists()
        and y_path.exists()
        and cat_features_path.exists()
        and embed_features_path.exists()
    ):
        print("Data is already prepared")
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        with open(cat_features_path, "rb") as f:
            cat_features = pickle.load(f)
        with open(embed_features_path, "rb") as f:
            embed_features = pickle.load(f)
        X = fill_missing_values(X, cat_features)
        X.info()
    else:
        if model_name in ["Ridge", "LightGBM"]:
            data = preprocess_data_ridge(train_path)
        elif model_name == "CatBoost":
            data = load_and_preprocess_data(
                model_name,
                train_path,
                text_embeddings_type=text_embeddings_type,
                image_embeddings_type=image_embeddings_type,
                use_reduced_rubert_embeddings=False,
                use_truncated_embeddings=use_truncated_embeddings,
                embed_add_as_separate_columns=embed_add_as_separate_columns,
                use_image_quality_features=use_image_quality_features,
            )
        X, y, cat_features, embed_features = (
            data["X"],
            data["y"],
            data["cat_features"],
            data["embed_features"],
        )

        # # save data to csv
        # X.to_csv(x_path, index=False)
        # y.to_csv(y_path, index=False)
        # with open(cat_features_path, "wb") as f:
        #     pickle.dump(cat_features, f)
        # with open(embed_features_path, "wb") as f:
        #     pickle.dump(embed_features, f)

    print("-------")
    from pprint import pprint

    # pprint(X.head(1))
    # pprint(y.head(1))
    pprint(X.info())
    pprint(y.info())
    pprint(f"cat_features: {cat_features}")
    pprint(f"embed_features: {embed_features}")
    print("-------")

    train_model(
        model_name, task_name, X, y, embed_features, use_stratified_kfold, use_hyperparameter_optimization
    )


if __name__ == "__main__":
    tyro.cli(main)
