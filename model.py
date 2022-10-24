# Model building and training
from re import T
import pandas as pd
import numpy as np
from torch import rand

# Modeling
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.model_selection import KFold


from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

from preprocessing import Preprocess


# importing data
import os

class Fruit_model:

    def __init__(self, prep: Preprocess, epochs= 40, batch_size=128, verbose=2, test_size=0.1, random_state=6, n=-1, lr=0.0001, momentum = .9):
        self.prep = prep
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.test_size = test_size
        self.random_state = random_state
        self.n = n
        self.lr = lr
        self.momentum = momentum




        self.shape = self.prep.X_train.shape[1:]
        if len(shape) < 3:
            shape = (shape[0], shape[1], 1)


    def define_model(self):
        # grab the pre-trained VGG19 model, removing the top layers and changing the input shape
        vgg_model = VGG19(weights="imagenet", include_top=False, input_shape= self.shape)

        # freeze pre-trained layers
        for layer in vgg_model.layers:
            layer.trainable = False

        # add new classifier layers
        flat1 = Flatten()(vgg_model.layers[-1].output)
        class1 = Dense(512, activation='relu')(flat1)
        drop1 = Dropout(0.5)(class1)
        class2 = Dense(512, activation='relu')(drop1)
        norm = BatchNormalization()(class2)
        drop2 = Dropout(.5)(norm)
        


        output = Dense(len(self.prep.classes), activation='softmax')(drop2)

        # define new model with top layers
        model = Model(inputs=vgg_model.inputs, outputs=output)
        opt = SGD(lr=self.lr, momentum = self.momentum)

        # es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # for output while training

        
        model.summary()

        return model

    
