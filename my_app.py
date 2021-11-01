

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


def image_classifier_model(img, model_file) -> float:
    """Returns the prediction score of image by the model Used"""

    model = load_model(model_file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    return np.argmax(prediction)


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

    label = image_classifier_model(
        img=upload_file, model_file='model/keras_model.h5'
    )

    st.write(label)
