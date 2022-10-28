# imports 
# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import interfaces

import json
import os

from skimage import io
import skimage
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf

from keras.utils import np_utils
import tensorflow as tf
import keras

import glob
import os

import random


class Preprocess(interfaces.Preprocessing_interface):
    """
    Preprocesses images for training, testing, validation, and unseen data sets 

    CONSTANTS:
    background_thresh --> 0 < x < 1 higher means more background will be whited out
    white_thres --> 0 < x < 1 this is the percentage of the image that is white if the image is above x percentage,
                    the image is reverted back to it's original form. Sometimes the background removal whites an entire image.
    
    For Training:
        Execute the 'prep_training' method on the desired training directory to populate the testing 
        and training variables. Pass the object into the training method in the Model_interface
    
    For Evaluation:
        Execute the 'prep_validation' method on the desired validation directory, it will populate the 
        necessary varaibles to use in the evaluate_model function in a Model_interface

    """

    def __init__(self, random_state = 6):


        self.classes = None
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_val = None
        self.y_val = None

        self.label_encoder = LabelEncoder()
        
        

    def run(self, img,  background_thresh = .5, white_thresh = .7):
        """
        Runs preprocessing for 1 image
        """

        segmentor = SelfiSegmentation()

        #resize office to 640x480
        img = cv2.resize(img, (150, 150))

        white = (255, 255, 255)

        imgNoBg = segmentor.removeBG(img, white, threshold= background_thresh)
        
        white_percentage = np.sum(imgNoBg == 255)/(np.sum(imgNoBg == 255) + np.sum(imgNoBg != 255))   
        
        if  white_percentage > white_thresh:
            return img

        return imgNoBg

    def prep_training(self, dir: str, test_size: float = .2):
        """
        Splits the preprocessed images into testing and training sets ready to be put into the model
        populates the X_train, X_test, y_train, y_test global varaibles
        """
        self.classes = os.listdir(dir)

        self.classes = [i.lower() for i in self.classes]


        X, y = self.run_for_set(dir)

        y = np.array([i.lower() for i in y])

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = self.random_state)

        # One hot encoding
        # encoding labels
        
        self.label_encoder.fit(y)

        vec_train = self.label_encoder.transform(y_train)
        vec_test = self.label_encoder.transform(y_test)

        # one hot encode target values
        self.y_train = to_categorical(vec_train)
        self.y_test = to_categorical(vec_test)

    def prep_validation(self, dir: str) -> list:
        """
        preprocess an entire set of images in a given directory "dir" 
        """

        self.X_val, y = self.run_for_set(dir)

        y = np.array([i.replace('stawberries', 'strawberries') for i in y])

        vec = self.label_encoder.transform(y)

        # one hot encode target values
        self.y_val = to_categorical(vec)

    
    def run_for_set(self, dir):
        """
        Returns preprocessed images from a given directory and their classes in a tuple
        """

        X = []
        y = []

        for path in glob.glob(dir + "\\*\\*"):
            
            image = io.imread(path)
            
            try:
                preprocessed = self.run(image)
                X.append(preprocessed)
                y.append(path.split('\\')[-2])
                
            except:
                pass

        return np.array(X), np.array(y)
    
    def save(self, dir):
        
       
        if not os.path.exists(dir):
            os.makedirs(dir)
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

    def clear(self):
        """Clears the object"""
        self.classes = None
        self.random_state = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_val = None
        self.y_val = None

        self.label_encoder = None
            

            

       




# prep = Preprocess()

# prep.run_for_set(prep.data_path + "\\train")
# prep.split_and_save(prep.data_path + "\\preprocessed")