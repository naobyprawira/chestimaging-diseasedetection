import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from keras.applications import DenseNet121
from keras.layers import Input, Dropout, Dense
from keras.models import Model

# # Define the class names
CLASS_NAMES = ['COVID-19','Healthy', 'Tuberculosis']

# Define model parameters
input_shape = (224, 224, 3)
n_classes = 3
final_activation = 'softmax'

def load_DenseNet121():
    input_tensor = Input(shape=input_shape)
    baseModel = DenseNet121(pooling='avg', include_top=False, input_tensor=input_tensor)
    model = FCLayers(baseModel)
    return model

def FCLayers(baseModel):
    baseModel.trainable = True
    headModel = baseModel.output
    headModel = Dropout(0.5, seed=73)(headModel)
    headModel = Dense(n_classes, activation=final_activation)(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    return model

model = load_DenseNet121()
model.load_weights('trained_model/DenseNet121_Weights.h5')

def import_and_predict(image_data):
    """Preprocess and predict the class of the input image."""
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction, CLASS_NAMES[np.argmax(prediction,)]





