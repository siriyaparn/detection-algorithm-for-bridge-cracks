# Detection Algorithm for Bridge Cracks Based on Deep Learning
Link : https://www.kaggle.com/code/siriyasriboonma/deep-learning-cnn

## Introduction
This project aims to detect bridge cracks surface as normal and crack surface based on deep learning, Convolutional Neural Network, using TensorFlow Keras. The dataset is from https://github.com/Charmve/Surface-Defect-Detection/tree/master/Bridge_Crack_Image.

This dataset is divided into two part:
1.	Train dataset contains 50,000 images of both normal and crack surface images. The label of this dataset is provided in train.txt file.
2.	Validation dataset contains 5,000 images of both normal and crack surface images. The label of this dataset is provided in val.txt file.
Since there is no test dataset, this project will use validation dataset as test data and use 20% of train dataset to be validation dataset.

## Import Libraries
The libraries which have been used in this project are shown as below.
```sh
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

from PIL import Image

import seaborn as sns
import plotly.express as px

from pathlib import Path

import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report
```

## Data Preprocessing

## Modeling Process
1. Parameter Setting : I set the parameter image_width and image_height = 16 equal to the image size which is 16x16. The image_color_channel_size = 255 due to RGB. 
```sh
## image_widht
image_widht = 16
## image_height
image_height = 16
## image_color_channel_size
image_color_channel_size = 255
## image_size
image_size = (image_widht, image_height)
## batch_size
batch_size = 8
## epochs
epochs = 20
## learning_rate
learning_rate = 0.01
## class_names
class_names = ['Normal','Crack']
```
2. Loading Image and Rescaling : The tf.keras.preprocessing.image.ImageDataGenerator is used to convert image to array and rescale image. For train_gen, I set validation_split = 0.2 which is 20% of 50,000 images.
```sh
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./image_color_channel_size,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./image_color_channel_size
)
```
Then, I load images from path in train_df and test_df the I have crated by using ImageDataGenerator.flow_from_dataframe. The parameter class_mode = binary because there are only two labels in this project which are ‘Normal’ and ‘Crack’. This can separate data into train, validation, and test. 
```sh
train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False,
    #seed=42
)
```
