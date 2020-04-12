import tensorflow as tf
from keras import layers, models, optimizers
from gait_generator import DataGenerator
from gait_json import create_json
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

path = '/home/liuz/detectron2/gait_dataset/'

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    parser = argparse.ArgumentParser(description="model stucture.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")               
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='gait_profile')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # parameters
    params = {'dim': (50,299,299),
              'batch_size': 1,
              'n_classes': 17,
              'n_channels': 3,
              'shuffle': True}   

    # datasets
    partition, labels = create_json()
    #print(labels)

    # generators
    training_generator = DataGenerator(list_IDs = partition['train'], labels = labels, **params)
    validation_generator = DataGenerator(list_IDs = partition['validation'], labels = labels, **params)    

    frames, rows, columns, channels = 50, 299, 299, 3
    classes = 17
    
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
    model.summary()
    
    # train
    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])
    
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    #EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_categorical_accuracy', mode='max',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))    
    #model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=[log, tb, checkpoint], validation_data=(x_valid, y_valid))
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        steps_per_epoch=165, #steps_per_epoch = int(number_of_train_samples / batch_size)
                        validation_steps=17,  #val_steps = int(number_of_val_samples / batch_size)
                        epochs=args.epochs, verbose=1,
                        callbacks=[log, tb, checkpoint, lr_decay])  
    
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)    