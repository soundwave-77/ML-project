import io
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from tqdm import tqdm
from clearml import Task


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def preprocess_data(df_train):
    print('==== Preparing Data ====')

    df_train['price'] = df_train['price'].fillna(df_train['price'].mean())

    for col in ['param_1', 'param_2', 'param_3', 'image_top_1', 'title', 'description']:
        df_train[col] = df_train[col].fillna('')

    df_train['image_top_1'] = df_train['image_top_1'].astype('str')

    df_train.drop(['image', 'item_id', 'user_id', 'activation_date', 'title', 'description'], axis=1, inplace=True)

    X = df_train.drop(columns=['deal_probability'])
    y = df_train['deal_probability']

    print('==== Data preprocessed successfully! ====')

    return X, y


def train_model(model_name, task_name, X, y, cat_features):
    # Initialize ClearML task
    task = Task.init(project_name='avito_sales_prediction', task_name=task_name)

    task.connect({
        'model_name': model_name,
        'dataset_size': X.shape,
        'numerical_features': X.select_dtypes(include='number').columns.to_list(),
        'categorical_features': cat_features
    })

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

        model = CatBoostRegressor(task_type='GPU' if is_gpu_available() else 'CPU',
                                  metric_period=50,
                                  early_stopping_rounds=5)
        model.fit(train_pool, eval_set=eval_pool)

        preds = model.predict(X_val)
        score = rmse(y_val, preds)
        scores.append(score)

    mean_score = np.mean(scores)

    print(f'Mean RMSE across folds: {mean_score}')
    task.connect({'mean_rmse_across_folds': mean_score})

    # Train model with best hyperparameters
    model = CatBoostRegressor(
        task_type='GPU' if is_gpu_available() else 'CPU',
        metric_period=50,
        early_stopping_rounds=5)
    model.fit(X, y, cat_features=cat_features)

    params = model.get_params()
    # Log hyperparameters to ClearML
    task.connect(params)

    # Evaluate final model
    preds = model.predict(X)
    rmse_score = rmse(preds, y)
    task.connect({'train_rmse': rmse_score})

    save_feature_importance(model, model_name, X, task)
    return model, rmse_score


def is_gpu_available():
    try:
        gpu_count = get_gpu_device_count()
        return gpu_count > 0
    except Exception:
        return False


def save_feature_importance(model, model_name, X, task):
    feature_importances = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by="importance", ascending=False)

    # Log the CSV to ClearML
    task.upload_artifact(name='Feature importance', artifact_object=importance_df.to_csv(index=False))
    print('---Feature importance logged to ClearML---')

    top_features = importance_df.head(50)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(20, 15))
    plt.barh(top_features['feature'], top_features['importance'], color='orange')
    plt.xlabel('Feature Importance')
    plt.title(f'Top Feature Importances for {model_name} Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf).convert('RGB')

    task.get_logger().report_image(
        title='Top Feature Importance',
        series='Top Feature Importance',
        iteration=0,
        image=image
    )
    print('---Top feature importance plot logged to ClearML---')
    buf.close()


def main(df_train, model_name, task_name):
    cat_features = ['region', 'city', 'parent_category_name', 'category_name', 
                    'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']

    # Preprocess data
    X, y = preprocess_data(df_train)

    # Train the model with hyperparameter optimization
    model, rmse = train_model(model_name, task_name, X, y, cat_features)
    print(f'Train RMSE: {rmse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on provided dataset')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset CSV file')
    parser.add_argument('--model_name', type=str, default='CatBoost', help='Name of the model to be trained')
    parser.add_argument('--task_name', type=str, default='CatBoost_Baseline', help='Name of the ClearML task')

    args = parser.parse_args()

    train_path = args.train_path
    model_name = args.model_name
    task_name = args.task_name

    df_train = pd.read_csv(train_path)

    main(df_train, model_name, task_name)