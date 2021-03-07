import numpy as np
from tensorflow import keras
import tensorflow as tf

class AsymmetricLoss(keras.losses.Loss):
    def __init__(self, shifting_ = 0.05, gamma_neg = 4, gamma_pos = 1):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.shift = shifting_
        


    def call(self, y_true, y_pred):
        pass