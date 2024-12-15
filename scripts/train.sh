# train baseline
python scripts/train_catboost.py --train_path data/raw/train.csv --task_name CatBoost_Baseline


# add text features
python scripts/train_catboost.py --train_path data/raw/train.csv --task_name catboost_rubert_features
