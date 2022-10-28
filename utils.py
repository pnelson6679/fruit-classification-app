# put all utility functions here

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Utils:

    @staticmethod
    def summarize_diagnostics(history):
        '''Plots diganostic learning curves of the model 
        Inputs: - HISTORY object generated during model fitting that maintains loss values and relevant metrics
                - TESTX and TESTY image arrays used for validating accuracy
        Returns: - Two plots (cross entropy loss and validation accuracy) over the number of epochs.  
        '''
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()