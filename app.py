

import streamlit as st
from PIL import Image
import image_classifier


st.title(
    "CT SCAN Image Classifier for Demented Brains with High Risk of Alzheimer's Disease"
)
st.header(
    "ALZHEIMER_CT_SCAN_IMAGE_CLASSIFIER"
)

upload_file = st.file_uploader("Upload CT BRAIN SCAN Here", type="jpg")

if upload_file is not None:
    upload_image = Image.open(upload_file)
    st.image(
        upload_image, caption='Uploaded Scan', use_column_width=True
    )

    st.write("")
    st.write("Running Model Algorithms ... ")

    label = image_classifier.image_classifier_model(
        img=upload_file, model_file='model/keras_model.h5'
    )

    st.write(label)
