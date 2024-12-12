# file receives a csv file and trains a catboost model on it
# file should have preprocessed data
# it also logs the model to clearml
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from tqdm import tqdm
import optuna
from clearml import Task
import json
import pickle


# NLTK stopwords download
nltk.download('stopwords')


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def load_embeddings(filepath: str):
    """
        Returns an array with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/resnet/embeddings_train_merged.npz
    """
    embed = np.load(filepath)
    return embed  # keys: ['embeddings', 'images'] 


def load_text_embeddings(filepath: str):
    """
        Returns a dictionary with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.json
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def load_text_pickle(filepath: str):
    """
        Returns a dictionary with keys as item_ids and values as embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.pkl
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_text_parquet(filepath: str):
    """
        Returns a dataframe with index as item_id and column for embeddings

        ~/Yandex.Disk/hse_ml_avito/vector_store/rubert_tiny_turbo/title_embeddings_reduced_train.parquet
    """
    df = pd.read_parquet(filepath)
    return df


def load_and_preprocess_data(train_path, test_path):
    stopWords = set(stopwords.words('russian'))
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("==== Preparing Data ====")

    vectorizer = TfidfVectorizer(stop_words=list(stopWords), max_features=2000)
    vectorizer.fit(df_train['title'])

    train_title = vectorizer.transform(df_train['title'])
    test_title = vectorizer.transform(df_test['title'])

    train_title_df = pd.DataFrame.sparse.from_spmatrix(train_title, columns=vectorizer.get_feature_names_out())
    test_title_df = pd.DataFrame.sparse.from_spmatrix(test_title, columns=vectorizer.get_feature_names_out())

    df_train['price'] = df_train['price'].fillna(df_train['price'].mean())
    df_test['price'] = df_test['price'].fillna(df_train['price'].mean())

    for col in ['param_1', 'param_2', 'param_3', 'image_top_1', 'title', 'description']:
        df_train[col] = df_train[col].fillna('')
        df_test[col] = df_test[col].fillna('')

    df_train['image_top_1'] = df_train['image_top_1'].astype('str')
    df_test['image_top_1'] = df_test['image_top_1'].astype('str')

    cat_features = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']
    text_features = ['title', 'description']

    df_train.drop(['image', 'item_id', 'user_id', 'activation_date'], axis=1, inplace=True)
    df_test.drop(['image', 'item_id', 'user_id', 'activation_date'], axis=1, inplace=True)

    del_cols = ['deal_prob_cat', 'params', 'price_log']

    for col in del_cols:
        if col in df_train.columns:
            df_train = df_train.drop(columns=col)
        if col in df_test.columns:
            df_test = df_test.drop(columns=col)

    X = df_train.drop(columns=['deal_probability', 'title', 'description'])
    # X_test = df_test.drop(columns=['title', 'description'])

    y = df_train['deal_probability']

    print("==== Data preprocessed successfully! ====")

    return X, y, cat_features


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


def is_gpu_available():
    try:
        gpu_count = get_gpu_device_count()
        return gpu_count > 0
    except Exception:
        return False


def main():
    # File paths
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    # Load and preprocess data
    X, y, cat_features = load_and_preprocess_data(train_path, test_path)

    # Train the model with hyperparameter optimization
    final_model, final_rmse = train_model(X, y, cat_features, n_trials=3)
    print(f'RMSE on Training Data: {final_rmse}')


if __name__ == '__main__':
    main()
