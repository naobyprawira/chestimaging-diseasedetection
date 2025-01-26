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

# # Define the class names
CLASS_NAMES = ['Non-TB','Healthy', 'Tuberculosis']

# # Define model parameters
# input_shape = (224, 224, 3)
# n_classes = 3
# final_activation = 'softmax'
# def Inception(X, p1, p2, p3, p4):
#     # Path 1 is a  1x1 convolutional layer
#     p1_1 = Conv2D(p1, kernel_size=(1,1), activation='relu')(X)
#     # Path 2 is a 1x1 convolutional layer followed by a 3x3 convolutional layer
#     p2_1 = Conv2D(p2[0], 1, 1, activation='relu')(X)
#     p2_2 = Conv2D(p2[1], 3, 1, padding='same', activation='relu')(p2_1)
#     # Path 3 is a 1x1 convolutional layer followed by a 5x5 convolutional layer
#     p3_1 = Conv2D(p3[0], 1, 1, activation='relu')(X)
#     p3_2 = Conv2D(p3[1], 5, 1, padding='same', activation='relu')(p3_1)
#     # Path 4 is a 3x3 max pooling layer followed by a 1x1 convolutional layer
#     p4_1 = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
#     p4_2 = Conv2D(p4, 1, 1, activation='relu')(p4_1)
#     # Concatenate the outputs on the channel dimension
#     return Concatenate(axis=-1)([p1_1, p2_2, p3_2, p4_2])

# def GoogLeNet(shape=(256, 256, 3), classes=3):

#     X_input = Input(shape)
#     X = Conv2D(64, kernel_size=7, strides=2, padding='same')(X_input)
#     X = BatchNormalization()(X) # Regularization technique
#     X = Activation('relu')(X)
#     X = MaxPool2D(pool_size=3, strides=2, padding='same')(X)
    
#     X = Conv2D(64, kernel_size=1, strides=1, padding='same')(X)
#     X = Conv2D(192, kernel_size=3, strides=1, padding='same')(X)
#     X = BatchNormalization()(X) # Regularization technique
#     X = Activation('relu')(X)
#     X = MaxPool2D(pool_size=3, strides=2, padding='same')(X)
    
#     X = Inception(X, 64, (96,128), (16,32), 32)
#     X = Inception(X, 128, (128,192), (32,96), 64)
#     X = MaxPool2D(pool_size=3, strides=2, padding='same')(X)
#     X = Inception(X, 192, (96, 208), (16, 48), 64)
#     X = Inception(X, 160, (112, 224), (24, 64), 64)
#     X = Inception(X, 128, (128, 256), (24, 64), 64)
#     X = Inception(X, 112, (144, 288), (32, 64), 64)
#     X = Inception(X, 256, (160, 320), (32, 128), 128)
#     X = MaxPool2D(pool_size=3, strides=2, padding='same')(X)
#     X = Inception(X, 256, (160, 320), (32, 128), 128)
#     X = Inception(X, 384, (192, 384), (48, 128), 128)
#     X = GlobalAveragePooling2D()(X)
#     X = Dropout(0.4)(X)
#     X = Dense(classes, activation='softmax')(X)
#     model = Model(inputs=X_input, outputs=X, name="GoogLeNet")
#     return model
# model= GoogLeNet()

model= load_model('trained_model/GoogleNet.h5')

# def load_DenseNet169():
#     input_tensor = Input(shape=input_shape)
#     baseModel = DenseNet169(pooling='avg', include_top=False, input_tensor=input_tensor)
#     model = FCLayers(baseModel)
#     return model

# def FCLayers(baseModel):
#     baseModel.trainable = True
#     headModel = baseModel.output
#     headModel = Dropout(0.5, seed=73)(headModel)
#     headModel = Dense(n_classes, activation=final_activation)(headModel)
#     model = Model(inputs=baseModel.input, outputs=headModel)
#     return model

# model = load_DenseNet169()
# model.load_weights('trained_model/GoogleNet.h5')

def import_and_predict(image_data):
    """Preprocess and predict the class of the input image."""
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction, CLASS_NAMES[np.argmax(prediction,)]