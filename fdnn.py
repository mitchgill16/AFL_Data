#fdnn model, extract features from best xgb_model, creatle new dl_model
#feature extractor model creator
import xgboost as xgb
import tensorflow as tf; print(tf.__version__)
import keras; print(keras.__version__)
#import torch.nn as nn
#import touch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint
from Gather_AFL_Data import gatherer as gad
import skopt
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Activation
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score, KFold

def load_features(xgb_model):
    #do feature extraction
    return features

def create_DNN(features):
    #do dnn
    return dnn

def create_CNN(features):
    #do cnn
    return cnn

def test():
    print("fdnn class works")

def main():
    best_xgb = pickle.load(open("best_accuracy_xgb.dat", "rb"))


if __name__ == '__main__':
    main()
