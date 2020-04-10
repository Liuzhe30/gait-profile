import tensorflow as tf
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.callbacks import Callback
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
import numpy as np
import os
import argparse
from keras import callbacks
from keras import backend as K 
K.clear_session()
K.set_image_data_format('channels_last')

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # load data
    x_train = np.load('gait_dataset/x_train_0.npy') #(165, 50, 299, 299, 3)
    y_train = np.load('gait_dataset/y_train_0.npy') #(165,)
    x_valid = np.load('gait_dataset/x_valid_0.npy') #(17, 50, 299, 299, 3)
    y_valid = np.load('gait_dataset/y_valid_0.npy') #(17,)  
    x_test = np.load('gait_dataset/x_test_0.npy') #(17, 50, 299, 299, 3)
    y_test = np.load('gait_dataset/y_test_0.npy') #(17,)  
    
    # build model
    video = Input(shape=(frames, rows, columns, channels))
    cnn_base = VGG16(input_shape=(rows, columns, channels), weights="imagenet", include_top=False)
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(input=cnn_base.input, output=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(256)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
    outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    
    # train
    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])