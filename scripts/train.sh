export ENABLE_CLEARML=1

python scripts/train.py \
    --train_path data/raw/train.csv \
    --task_name clip_embed \
    --model_name CatBoost \
    --use_stratified_kfold False \
    --embed_add_as_separate_columns True \
    --add_image_features True \
    --use_prep_data_cache False \
    --use_truncated_embeddings True
