# catboost
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name catboost_rubert_features_title_description \
#     --model_name CatBoost

# ridge
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name ridge_baseline \
#     --model_name Ridge

# lightgbm
python scripts/train.py \
    --train_path data/raw/train.csv \
    --task_name lightgbm_baseline \
    --model_name LightGBM
