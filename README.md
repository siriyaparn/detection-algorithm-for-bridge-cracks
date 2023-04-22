# Detection Algorithm for Bridge Cracks Based on Deep Learning
Link : https://www.kaggle.com/code/siriyasriboonma/deep-learning-cnn

## Introduction
This project aims to detect bridge cracks surface as normal and crack surface based on deep learning, Convolutional Neural Network, using TensorFlow Keras. The dataset is from https://github.com/Charmve/Surface-Defect-Detection/tree/master/Bridge_Crack_Image.

This dataset is divided into two part:
1.	Train dataset contains 50,000 images of both normal and crack surface images. The label of this dataset is provided in train.txt file.
2.	Validation dataset contains 5,000 images of both normal and crack surface images. The label of this dataset is provided in val.txt file.
Since there is no test dataset, this project will use validation dataset as test data and use 20% of train dataset to be validation dataset.
