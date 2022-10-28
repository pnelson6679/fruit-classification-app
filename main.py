# imports

from model import Model_vgg19_small
from preprocessing import Preprocess

import json

with open('config.json', 'r') as file:
    config = json.load(file)

import logging

data_path = config['data_path'] + '\\MY_data'

# getting training and validation sets preprocessed
prep = Preprocess()

logging.info("Preprocessing training data...")
prep.prep_training(data_path + "\\train")

logging.info("Preprocessing unseen validation set")
prep.prep_validation(data_path + "\\test")

# defining model
logging.info("Initializing model object and defining model architecture...")
fruit_model = Model_vgg19_small(prep)

fruit_model.define_model()

# training model
logging.info("Training model...")
fruit_model.train()

# # evaluating model
# logging.info('Evaluating model...')
# fruit_model.evaluate()

# saving weights
logging.info("Saving model...")
fruit_model.save()