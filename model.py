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

from datetime import datetime

from sklearn.model_selection import KFold


from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

from preprocessing import Preprocess

import interfaces

from utils import Utils

# importing data
import os

class Model_vgg19_small(interfaces.Model_interface):


    def __init__(self, prep: Preprocess):

        self.model = None
        self.prep = prep

        self.shape = self.prep.X_train.shape[1:]
        if len(self.shape) < 3:
            self.shape = (self.shape[0], self.shape[1], 1)
                

    def define_model(self, lr=0.0001, momentum = .9) -> None:
        """Define a model"""

        # defines the model structure

        print('Defining Model...')
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
        opt = SGD(learning_rate=lr, momentum = momentum)

        # es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # for output while training        
        model.summary()

        self.model = model
        

    def train(self, epochs = 5, batch_size = 8, verbose=2) -> None:
        """Trains the model
        The preprocessing_interface passed into the function must have run the 'split' function to populate the X_train, X_test,
        y_train, y_test object variables

        epochs= 15
        batch_size=128
        verbose=2
        test_size=0.1
        random_state=6
        shape = X_train.shape[1:]
        if len(shape) < 3:
            shape = (shape[0], shape[1], 1)
        n=-1
        
        
        """

        print('Fitting Model...')

        self.history = self.model.fit(self.prep.X_train, self.prep.y_train, epochs=epochs, batch_size=batch_size, \
            validation_data=(self.prep.X_test, self.prep.y_test), verbose=verbose)

        

        print('Model fitted! Epochs=%d, Batch Size=%d' % (epochs, batch_size))
        

    def evaluate(self) -> dict:
        """Evaluates model with plotting and returning accuracy and loss in a dictionary {loss: float, accuracy: decimal}
        Thsi preprocessing_interface MUST be different that the one used to train the dataset, otherwise we're evaluating the model based on the
        training data. 

        the preprocessing_interface object must have ran the 'run_set' function to create the validation set of data
        """

        
        
        print('Model Evalution on Unseen Validation Set:')
        results = self.model.evaluate(self.prep.X_val, self.prep.y_val)
        print("loss, acc:", results)

        # learning curves

        Utils.summarize_diagnostics(self.history)
        print('----FINISHED----')
        
        return {'loss': results[0], 'accuracy': results[1]}

    def predict(self, image) -> str:
        """Returns a prediction given a preprocessed image"""
        pass

    def save(self):
        """saves model artifacts"""
        self.model.save_weights('vgg_19_small\\' + str(datetime.today().year) + str(datetime.today().month) + str(datetime.today().day))

    def clear(self):
        """Clears object"""
        self.classes = None
        self.random_state = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_val = None
        self.y_val = None

        self.label_encoder = None

    def load(self, dir: str, segment: str):
        """
        NOT IMPLEMENTED

        Loads pickled data for a certain segment of model process
        segment = "train", "validation"

        with open(dir + '\\train.npy', 'rb') as f:
            np.save(f, self.X_train, allow_pickle = False)
            np.save(f, self.y_train, allow_pickle = False)
            np.save(f, self.classes, allow_pickle = False)

        with open(dir + '\\test.npy', 'rb') as f:    
            np.save(f, self.X_test, allow_pickle = False)
            np.save(f, self.y_test, allow_pickle = False)
            np.save(f, self.classes, allow_pickle = False)
            np.save(f, self.label_encoder, allow_pickle = False)

        with open(dir + '\\val.npy', 'rb') as f: 
            np.save(f, self.X_val, allow_pickle = False)
            np.save(f, self.y_val, allow_pickle = False)
            np.save(f, self.classes, allow_pickle = False)
            np.save(f, self.label_encoder, allow_pickle = False)
        """

        if segment == "train":
            with open(dir + "\\rain.npy", 'rb') as f:
                self.X_train = np.load(f)
                self.y_train = np.load(f)
                

            with open(dir + '\\test.npy', 'rb') as f:    
                self.X_test = np.load(f)
                self.y_test = np.load(f)
                self.classes = np.load(f)
                self.label_encoder = np.load(f)

        if segment == "val":
            with open(dir + '\\val.npy', 'rb') as f: 
                self.X_val = np.load(f)
                self.y_val = np.load(f)
                self.classes = np.load(f)
                self.label_encoder = np.load(f)


    
    
