
from h5py._hl.files import File
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np



def image_classifier_model(img: File, model_file: File) -> float:
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
