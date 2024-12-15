def is_gpu_available():
    try:
        gpu_count = get_gpu_device_count()
        return gpu_count > 0
    except Exception:
        return False



import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from tqdm import tqdm
import optuna

from prepare_data import load_and_preprocess_data


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_model(X, y, cat_features, n_trials=10):
    # Initialize ClearML task
    # task = Task.init(project_name='avito_sales_prediction', task_name='train_pipeline_with_optuna')

    def objective(trial):
        # Hyperparams for boosting models
        boosting_params = {
            # The number of trees in the model
            'n_estimators': trial.suggest_int(
                'n_estimators', 500, 1100, step=300
            ),
            # Maximum depth of trees
            'max_depth': trial.suggest_int(
                'max_depth', 3, 11
            ),
            # Learning rate
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-2, 1e-1
            ),
            # Task type (CPU or GPU)
            'task_type': 'GPU' if is_gpu_available() else 'CPU',
            # Iteration period of metric calculation
            'metric_period': 50,
            # Avoid overfitting if the metric does not change
            'early_stopping_rounds': 5,
        }

        # Cross-validation
        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratify_labels = (y.round(2) * 100).astype(int)
        FOLD_LIST = list(skf.split(stratify_labels, stratify_labels))

        for train_idx, val_idx in tqdm(FOLD_LIST):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            eval_pool = Pool(X_val, y_val, cat_features=cat_features)
        
            model = CatBoostRegressor(**boosting_params)
            model.fit(train_pool, eval_set=eval_pool)
        
            preds = model.predict(X_val)
            score = rmse(y_val, preds)
            scores.append(score)

        avg_score = np.mean(scores)
        return avg_score
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    # Log best hyperparameters to ClearML
    # task.connect(best_params)

    # Train final model with best hyperparameters
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X, y, cat_features=cat_features)
    
    # Evaluate final model
    preds_final = final_model.predict(X)
    final_rmse = rmse(preds_final, y)
    # task.connect({'final_rmse': final_rmse})

    final_model_path = 'models/catboost_model.cbm'
    final_model.save_model(final_model_path)

    # # Log the model to ClearML
    # task.upload_artifact(name='catboost_model', artifact_object='models/catboost_model.cbm')

    return final_model, final_rmse




def main():

    # Train the model with hyperparameter optimization
    final_model, final_rmse = train_model(X, y, cat_features, n_trials=3)
    print(f'RMSE on Training Data: {final_rmse}')


if __name__ == '__main__':
    main()
