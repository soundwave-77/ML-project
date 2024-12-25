import sys
from dataclasses import dataclass
from pathlib import Path
import logging
import clip
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool
from PIL import Image
import shap
from scripts.prepare_data import fill_missing_values, add_text_statistics, preprocess_data_for_model
from scripts.extract_image_statistics import process_image
from scripts.cos_sim import compute_title_description_sim
from streamlit_shap import st_shap

from dotenv import load_dotenv
import os
import yadisk
load_dotenv()
# sys.path.append("/home/qb/study/hse/ml/project/ML-project")


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
REMOTE_CKPT_PATH = "/hse_ml_avito/checkpoints/final_model.cbm"
REMOTE_ARTIFACTS_PATH = "/hse_ml_avito/demo"
LOCAL_ARTIFACTS_PATH = "artifacts"
Path(LOCAL_ARTIFACTS_PATH).mkdir(parents=True, exist_ok=True)


@st.cache_data
def get_artifacts():
    """Download artifacts from yandex disk"""
    # list of regions
    yd_token = os.getenv("YANDEX_DISK_TOKEN")
    client = yadisk.Client(token=yd_token)
    
    with client:
        file_names = [obj.name for obj in client.listdir(REMOTE_ARTIFACTS_PATH)]
        for file_name in file_names:
            client.download(REMOTE_ARTIFACTS_PATH + "/" + file_name, LOCAL_ARTIFACTS_PATH + "/" + file_name)

        # download catboost model
        client.download(REMOTE_CKPT_PATH, LOCAL_ARTIFACTS_PATH + "/final_model.cbm")

        # download tfidf vectorizer
        client.download("/hse_ml_avito/vector_store/tfidf/title_description_tfidf_svd_model.pkl", LOCAL_ARTIFACTS_PATH + "/title_description_tfidf_svd_model.pkl")

        # download tsvd image embeddings
        client.download("/hse_ml_avito/vector_store/clip/tsvd_img_embeddings.joblib", LOCAL_ARTIFACTS_PATH + "/tsvd_img_embeddings.joblib")


get_artifacts()


@st.cache_resource
def get_ctb_model():
    print("Loading Catboost Model")  # logging.debug
    # load model and cache it
    model = CatBoostRegressor()
    model.load_model(LOCAL_ARTIFACTS_PATH + "/final_model.cbm")
    return model


@st.cache_resource
def get_clip_model():
    logging.debug("Loading clip")
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


@st.cache_resource
def get_tfidf_vectorizer():
    print("Loading tfidf") # logging.debug
    # load vectorizer and cache it
    # path = Path("/home/qb/Yandex.Disk/hse_ml_avito/vector_store/tfidf/title_tfidf_svd_model.pkl")
    path = Path(
        LOCAL_ARTIFACTS_PATH + "/title_description_tfidf_svd_model.pkl"
    ).expanduser()
    return joblib.load(path)


def load_tsvd_img_embeddings():
    path = Path(
        LOCAL_ARTIFACTS_PATH + "/tsvd_img_embeddings.joblib"
    ).expanduser()
    return joblib.load(path)


def read_list(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


@st.cache_data
def get_region_list() -> list[str]:
    # load region list and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/region_list.txt")


@st.cache_data
def get_category_list() -> list[str]:
    # load category list and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/category_list.txt")


@st.cache_data
def get_image_classes() -> list[str]:
    # load image classes and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/image_class_list.txt")


@st.cache_data
def get_city_list() -> list[str]:
    # load city list and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/city_list.txt")


@st.cache_data
def get_parent_category_list() -> list[str]:
    # load parent category list and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/parent_category_list.txt")


@st.cache_data
def get_user_type_list() -> list[str]:
    # load user type list and cache it
    return read_list(LOCAL_ARTIFACTS_PATH + "/user_type_list.txt")


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
CAT_FEATURES = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
    "image_top_1",
]

# =========== Preprocessing Functions =========================

def preprocess_data(df, image):
    cat_features = list(df.select_dtypes(include="object").columns)
    df = fill_missing_values(df, cat_features)
    df = add_text_statistics(df)
    # add image statistics
    stats = process_image(None, image)

    for key in stats.keys():
        df[key] = stats[key]

    # obtain title embeddings
    title_embeddings = tfidf_vectorizer.transform(df["title"])
    description_embeddings = tfidf_vectorizer.transform(df["description"])

    # compute title-description cosine similarity
    cos_sim = compute_title_description_sim(title_embeddings, description_embeddings)

    df["title_description_cos_sim"] = cos_sim

    df = preprocess_data_for_model(df, model_name="CatBoost")

    # obtain image embeddings
    image_features = get_image_embedding(image)
    tsvd = load_tsvd_img_embeddings()
    image_features = tsvd.transform(image_features)
    embed_df = pd.DataFrame(image_features)
    embed_cols = [f"image_embedding_{i+1}" for i in range(embed_df.shape[1])]
    embed_df.columns = embed_cols
    df = df.merge(embed_df, left_index=True, right_index=True)

    embed_df = pd.DataFrame(title_embeddings)
    embed_cols = [f"title_embedding_{i}" for i in range(embed_df.shape[1])]
    embed_df.columns = embed_cols
    df = df.merge(embed_df, left_index=True, right_index=True)

    embed_df = pd.DataFrame(description_embeddings)
    embed_cols = [f"description_embedding_{i}" for i in range(embed_df.shape[1])]
    embed_df.columns = embed_cols
    df = df.merge(embed_df, left_index=True, right_index=True)

    # st.write(df.columns)
    df.drop(columns=["title", "description", "image"], inplace=True)
    # cat_features = list(df.select_dtypes(include="object").columns)
    pool = Pool(df, cat_features=CAT_FEATURES)
    return pool, df


def predict(df, image):
    # Preprocess the data
    X, df = preprocess_data(df, image)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return predictions, shap_values, X, df, explainer


# ======= App Logic =======

st.title("Оценить товар по тексту и фотографиям")
st.write(
    "Загрузите фотографии товара и введите описание, чтобы получить оценку для товара."
)

title_input = st.text_input("Название товара")
text_input = st.text_area("Описание товара")
price_input = st.number_input("Цена", min_value=0, max_value=10_000_000, value=0)

with st.expander("Дополнительные параметры"):
    region_input = st.selectbox("Регион", region_list)
    city_input = st.selectbox("Город", city_list)
    parent_category_input = st.selectbox("Родительская категория", parent_category_list)
    category_input = st.selectbox("Категория", category_list)
    param1_input = st.number_input("Параметр 1", min_value=0, max_value=100, value=0)
    param2_input = st.number_input("Параметр 2", min_value=0, max_value=100, value=0)
    param3_input = st.number_input("Параметр 3", min_value=0, max_value=100, value=0)
    item_seq_input = st.number_input(
        "Порядковый номер товара у продавца", min_value=0, max_value=100, value=1
    )
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
                "image_top_1": [image_top1_input],
                "title": [title_input],
                "description": [text_input],
            }
        )
        predictions, shap_values, X, X_display, explainer = predict(df, image)

        st.success(f"Предсказанная вероятность продажи: {predictions[0]:.2f}")

        # Display the uploaded image
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        # Generate and display the SHAP summary plot
        st.subheader("Объяснение предсказаний модели")

        # fig, ax = plt.subplots(figsize=(10, 6))
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0, ...], X_display.iloc[0, :]))
        # shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type="bar", show=False)
        # st.pyplot(fig)
    else:
        st.error("Заполните все поля")
