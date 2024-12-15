from catboost import Pool, CatBoostClassifier


train_pool = Pool(data=X_train, label=y_train, text_features=['review'])
test_pool = Pool(data=X_test, label=y_test, text_features=['review'])


print('Train dataset shape: {}\n'.format(train_pool.shape))

def fit_model(train_pool, test_pool, **kwargs):
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        eval_metric='AUC',
        **kwargs
    )

    return model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=100,
    )


if __name__ == '__main__':
    model = fit_model(train_pool, test_pool, task_type='GPU')