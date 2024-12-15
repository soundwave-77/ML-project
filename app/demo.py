import streamlit as st
from PIL import Image


def predict(text, images):
    return f"Ad rating: {len(images) * 2}/10 (dummy)"


st.set_page_config(
    page_title="Ad Rating App", page_icon=":shaved_ice:", layout="centered"
)
st.title("Rate an Ad by Text and Images")
st.write("Upload product photos and enter a description to get a rating for the ad.")

text_input = st.text_area("Product Description")

image_inputs = st.file_uploader(
    "Upload Product Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if st.button("Rate Ad"):
    if text_input and image_inputs:
        images = [Image.open(image_input) for image_input in image_inputs]

        result = predict(text_input, images)

        st.success(result)

        for image in images:
            st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.error("Please fill in all fields.")
