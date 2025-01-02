import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model (Ensure the path is correct)
MODEL_PATH = './result/GoogLeNet_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class names
CLASS_NAMES = ['Healthy', 'Pneumonia', 'COVID-19']

def import_and_predict(image_data):
    """Preprocess and predict the class of the input image."""
    size = (224, 224)  # Resize to match the model input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.asarray(image) / 255.0  # Normalize the image
    img_reshape = img_array[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction, CLASS_NAMES[np.argmax(prediction)]
