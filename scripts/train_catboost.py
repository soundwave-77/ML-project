# file receives a csv file and trains a catboost model on it
# file should have preprocessed data
# it also logs the model to clearml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


stopWords = set(stopwords.words('russian'))
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

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
X_test = df_test.drop(columns=['title', 'description'])

y = df_train['deal_probability']

print("======= K-Fold Training ========")

spliter = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=3)

_y = (df_train.deal_probability.round(2)*100).astype(int)

FOLD_LIST = list(spliter.split(_y, _y))



_models = []

oof_predictions = np.zeros(shape=[X.shape[0]])

for fold_id, (train_idx, val_idx) in tqdm(enumerate(FOLD_LIST)):
    
    X_train, Y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, Y_val = X.loc[val_idx], y.loc[val_idx]
    
    train_dataset = Pool(X_train, Y_train,
                         cat_features=cat_features)
    
    eval_dataset = Pool(X_val, Y_val,
                        cat_features=cat_features)
    
    model = CatBoostRegressor(
        learning_rate=0.1, iterations=1000, eval_metric='RMSE',
        metric_period=50, early_stopping_rounds=20, task_type="GPU",
    )
    model.fit(train_dataset, eval_set=eval_dataset)
    
    _models.append(model)
    preds = model.predict(X_val)
    oof_predictions[val_idx] += preds
    print('fold_id:', fold_id, rmse(Y_val, preds))


pred = model.predict(X_test)

deal_probability = np.clip(pred, 0, 1)
# sample_sub.to_csv('sub.csv', index=False)


print("======= Report Metrics ========")

print(f"RMSE: {rmse(y, oof_predictions)}")
