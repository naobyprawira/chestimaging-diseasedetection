import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, 
    BatchNormalization, MaxPool2D, GlobalAveragePooling2D, 
    Input, Concatenate
)
from tensorflow.keras.applications import DenseNet121

# # Define the class names
CLASS_NAMES = ['Non-TB','Healthy', 'Tuberculosis']

# Define model parameters
def FCLayers(baseModel):
    baseModel.trainable = True
    headModel = baseModel.output
    headModel = Dropout(0.5, seed=73)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)
    model = Model(inputs = baseModel.input, outputs = headModel)
    return model

def load_DenseNet121():
    baseModel = DenseNet121(pooling='avg',
                            include_top=False, 
                            input_shape=(256,256,3))
    
    model = FCLayers(baseModel)
    return model

model= load_DenseNet121()
model.load_weights('trained_model/DenseNet121.weights.h5')

def import_and_predict(image_data):
    """Preprocess and predict the class of the input image."""
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction, CLASS_NAMES[np.argmax(prediction,)]