import tensorflow as tf
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
# import cv2
from tensorflow.keras.preprocessing.image import load_img




from tensorflow import keras
from tensorflow.keras.utils import Sequence, to_categorical
# from imutils import paths
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import glob
# import pandas as pd
import shutil
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras_vggface.vggface import VGGFace

from tensorflow.keras.applications.resnet50 import preprocess_input
# train_data, val_data = loadData("train",image_path,labels, val_split=0.2)
# restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
baseModel = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg',model='resnet50')

# Xây thêm các layer
# Lấy output của ConvNet trong VGG16
fcHead = baseModel.output

# Flatten trước khi dùng FCs
fcHead = Flatten(name='flatten')(fcHead)

# Thêm FC
fcHead = Dense(256, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)

# Output layer với softmax activation
fcHead = Dense(429, activation='softmax')(fcHead)

# Xây dựng model bằng việc nối ConvNet của VGG16 và fcHead
model = model = Model(inputs=baseModel.input, outputs=fcHead)
model.load_weights('data_model/face_recognition_restnet50_train_val_test.h5')
model.summary()
