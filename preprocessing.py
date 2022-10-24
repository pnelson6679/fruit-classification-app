# imports 
# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import json

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

with open('config.json', 'r') as file:
    config = json.load(file)




class Preprocess:
    """
    Preprocesses images for training, testing, validation, and unseen data sets 

    CONSTANTS:
    background_thresh --> 0 < x < 1 higher means more background will be whited out
    white_thres --> 0 < x < 1 this is the percentage of the image that is white if the image is above x percentage,
                    the image is reverted back to it's original form. Sometimes the background removal whites an entire image.

    """

    def __init__(self, random_state = 6):

        self.data_path = config['data_path'] + '\\MY_data'

        self.classes = os.listdir(self.data_path + "\\train")
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.label_encoder = None
        
        

    def run(self, img,  background_thresh = .3, white_thresh = .7):
        """
        Runs preprocessing for 1 image
        """

        segmentor = SelfiSegmentation()

        #resize office to 640x480
        img = cv2.resize(img, (640, 480))

        white = (255, 255, 255)

        imgNoBg = segmentor.removeBG(img, white, threshold= background_thresh)
        
        white_percentage = np.sum(imgNoBg == 255)/(np.sum(imgNoBg == 255) + np.sum(imgNoBg != 255))   
        
        if  white_percentage > white_thresh:
            return img

        return imgNoBg

    def run_for_set(self, dir):
        """
        preprocess an entire set of images in a given directory "dir" 
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

        self.X = X
        self.y = y

        return self.X, self.y

    def split(self, test_size = .2):

        X = np.array(self.X)
        y = np.array(self.y)
        
        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size= self.test_size, random_state = self.random_state)

        # One hot encoding
        # encoding labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        vec_train = self.label_encoder.transform(y_train)
        vec_test = self.label_encoder.transform(y_test)

        # saving validation labels for confusion matrix
        self.classes = set(y_test)

        # one hot encode target values
        self.y_train = to_categorical(vec_train)
        self.y_test = to_categorical(vec_test)

    def save(self, dir)
        
        with open(dir + '\\training.npy', 'rb') as f:
            np.save(f, self.X_train, allow_pickle = False)
            np.save(f, self.y_train, allow_pickle = False)
            np.save(f, self.X_test, allow_pickle = False)
            np.save(f, self.y_test, allow_pickle = False)
            np.save(f, self.classes, allow_pickle = False)




# prep = Preprocess()

# prep.run_for_set(prep.data_path + "\\train")
# prep.split_and_save(prep.data_path + "\\preprocessed")