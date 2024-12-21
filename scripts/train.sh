export ENABLE_CLEARML=1
export DEBUG=0

python scripts/train.py \
    --train_path data/raw/train.csv \
    --task_name baseline_title_fasttext \
    --model_name CatBoost \
    --use_stratified_kfold False \
    --use_hyperparameter_optimization True \
    --embed_add_as_separate_columns True \
    --use_prep_data_cache False \
    --use_truncated_embeddings True \
    --text_embeddings_type fasttext \
    --image_embeddings_type None \
    --use_image_quality_features False
