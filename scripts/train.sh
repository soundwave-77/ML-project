# catboost
python scripts/train.py \
    --train_path data/raw/train.csv \
    --task_name catboost_image_statistics \
    --model_name CatBoost \
    --text_embeddings_type tfidf
    --image_embeddings_type None

# ridge
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name ridge_baseline \
#     --model_name Ridge \
#     --text_embeddings_type tfidf
#     --image_embeddings_type None

# lightgbm
# python scripts/train.py \
#     --train_path data/raw/train.csv \
#     --task_name lightgbm_baseline \
#     --model_name LightGBM \
#     --text_embeddings_type tfidf
#     --image_embeddings_type None
