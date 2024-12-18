# catboost
export ENABLE_CLEARML=1
# export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/train.py \
    --train_path data/raw/train.csv \
    --task_name catboost_rubert_features_title_longer_sep_embed_no_stratified_kfold \
    --model_name CatBoost \
    --use_stratified_kfold False \
    --embed_add_as_separate_columns True

# catboost_rubert_features_title_longer_dim2 - storing embeddings in a single column

# ridge
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name ridge_baseline \
#     --model_name Ridge

# lightgbm
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name lightgbm_baseline \
#     --model_name LightGBM
