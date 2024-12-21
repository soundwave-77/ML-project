import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import shap
import matplotlib.pyplot as plt

sys.path.append("/home/qb/study/hse/ml/project/ML-project")


@dataclass
class Arguments:
    embed_add_as_separate_columns: bool = True


st.set_page_config(
    page_title="Оценить товар по тексту и фотографиям",
    page_icon=":shaved_ice:",
    layout="centered",
)

# =========== Preparing =========================
# TODO: later use relative path for artifacts
CKPT_PATH = Path("model.cbm")
SAVE_PATH = Path("outputs/demo")


@st.cache_data
def get_artifacts():
    """Download artifacts from yandex disk"""
    # list of regions
    # download catboost model
    # download tfidf vectorizer
    pass


get_artifacts()


@st.cache_resource
def get_ctb_model():
    # load model and cache it
    model = CatBoostRegressor()
    model.load_model(CKPT_PATH)
    return model


@st.cache_resource
def get_clip_model():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


@st.cache_resource
def get_tfidf_vectorizer():
    # load vectorizer and cache it
    # path = Path("/home/qb/Yandex.Disk/hse_ml_avito/vector_store/tfidf/title_tfidf_svd_model.pkl")
    path = Path(
        "/home/qb/Yandex.Disk/hse_ml_avito/vector_store/tfidf/title_description_tfidf_svd_model.pkl"
    )
    return joblib.load(path)


def read_list(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


@st.cache_data
def get_region_list() -> list[str]:
    # load region list and cache it
    return read_list(SAVE_PATH / "region_list.txt")


@st.cache_data
def get_category_list() -> list[str]:
    # load category list and cache it
    return read_list(SAVE_PATH / "category_list.txt")


@st.cache_data
def get_image_classes() -> list[str]:
    # load image classes and cache it
    return read_list(SAVE_PATH / "image_class_list.txt")


@st.cache_data
def get_city_list() -> list[str]:
    # load city list and cache it
    return read_list(SAVE_PATH / "city_list.txt")


@st.cache_data
def get_parent_category_list() -> list[str]:
    # load parent category list and cache it
    return read_list(SAVE_PATH / "parent_category_list.txt")


@st.cache_data
def get_user_type_list() -> list[str]:
    # load user type list and cache it
    return read_list(SAVE_PATH / "user_type_list.txt")


def get_image_embedding(image: Image) -> np.ndarray:
    """Obtain clip embedding for image"""
    clip_model, preprocess = get_clip_model()
    image = preprocess(image)
    image_features = clip_model.encode_image(image.unsqueeze(0))
    image_features = image_features.detach().numpy()
    return image_features


# initialize
model = get_ctb_model()
tfidf_vectorizer = get_tfidf_vectorizer()
region_list = get_region_list()
city_list = get_city_list()
parent_category_list = get_parent_category_list()
category_list = get_category_list()
image_classes = get_image_classes()
user_type_list = get_user_type_list()
args = Arguments()
cat_features = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
    "image_top1",
]

# =========== Preprocessing Functions =========================


def preprocess_data(df, image):
    # obtain text embeddings
    text_features = tfidf_vectorizer.transform(df["title"])

    # obtain image embeddings
    image_features = get_image_embedding(image)
    tsvd = TruncatedSVD(32)
    image_features = tsvd.fit_transform(image_features)

    if args.embed_add_as_separate_columns:
        embed_df = pd.DataFrame(text_features)
        embed_cols = [f"title_embedding_{i+1}" for i in range(embed_df.shape[1])]
        embed_df.columns = embed_cols
        df = df.merge(embed_df, left_index=True, right_index=True)

        # df["text_features"] = text_features
        # df["image_features"] = image_features
    else:
        df["combined_features"] = np.concatenate(
            [text_features, image_features], axis=1
        )
    st.write(df.columns)
    df.drop(columns=["title", "description"], inplace=True)

    pool = Pool(df, cat_features=cat_features)
    return pool


def predict(df, image):
    # Preprocess the data
    X = preprocess_data(df, image)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return predictions, shap_values, X


# ======= App Logic =======

st.title("Оценить товар по тексту и фотографиям")
st.write(
    "Загрузите фотографии товара и введите описание, чтобы получить оценку для товара."
)

title_input = st.text_input("Название товара")
text_input = st.text_area("Описание товара")
region_input = st.selectbox("Регион", region_list)
city_input = st.selectbox("Город", city_list)
parent_category_input = st.selectbox("Родительская категория", parent_category_list)
category_input = st.selectbox("Категория", category_list)
param1_input = st.number_input("Параметр 1", min_value=0, max_value=100, value=0)
param2_input = st.number_input("Параметр 2", min_value=0, max_value=100, value=0)
param3_input = st.number_input("Параметр 3", min_value=0, max_value=100, value=0)
item_seq_input = st.number_input(
    "Порядковый номер товара у продавца", min_value=0, max_value=100, value=0
)
price_input = st.number_input("Цена", min_value=0, max_value=10_000_000, value=0)
user_type_input = st.selectbox("Тип пользователя", ["Частное лицо", "Компания"])
image_top1_input = st.selectbox("Топ-1 изображение", image_classes)


image_input = st.file_uploader(
    "Загрузить фотографии товара",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if st.button("Оценить"):
    if text_input and image_input:
        image = Image.open(image_input)

        df = pd.DataFrame(
            {
                "region": [region_input],
                "city": [city_input],
                "parent_category_name": [parent_category_input],
                "category_name": [category_input],
                "param_1": [param1_input],
                "param_2": [param2_input],
                "param_3": [param3_input],
                "price": [price_input],
                "item_seq_number": [item_seq_input],
                "user_type": [user_type_input],
                "image_top1": [image_top1_input],
                "title": [title_input],
                "description": [text_input],
            }
        )
        predictions, shap_values, X = predict(df, image)

        st.success(f"Предсказанная цена: {predictions[0]} ₽")

        # Display the uploaded image
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        # Generate and display the SHAP summary plot
        st.subheader("Объяснение предсказаний модели")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False, ax=ax)
        st.pyplot(fig)
    else:
        st.error("Заполните все поля")
