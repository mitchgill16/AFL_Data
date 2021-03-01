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

def load_features(m):
    #loads features with gain values in prediction
    features = m.get_booster().get_score(importance_type="gain")
    #removes the f from these features
    new_dict = {}
    for key in features:
        new_key = int(key[1:])
        new_dict[new_key] = features[key]
    pd_f = pd.Series(new_dict)
    df = pd_f.to_frame()
    df = df.rename(columns = {0:'F_Score(GAIN)'})
    #sorts features by 250 most important
    indexes = df.nlargest(len(df), "F_Score(GAIN)").index
    indexes = indexes.sort_values()
    return indexes

def create_DNN(x_len):
    model = Sequential()
    model.add(Dense(250, input_dim = x_len))
    model.add(Activation('relu'))
    model.add(Dropout(0.03))
    model.add(BatchNormalization())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.02))

    model.add(Dense(64))
    model.add(Activation('relu'))

    x = 1
    while x<51:
        model.add(Dense(32))
        model.add(Activation('relu'))
        x = x + 1

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(8))
    model.add(Activation('relu'))
    #add output layer
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adamax(learning_rate=0.009)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    print(model.summary())
    return model

def create_CNN(x_len):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=14,
                     input_shape=(x_len, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=16, kernel_size=10,
                     input_shape=(32, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=10, kernel_size=8,
                     input_shape=(16, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adamax(learning_rate=0.003)#, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax"

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    print(model.summary())

def ohe_data(x_data, enc, flag):
    if (flag == 0):
        #transposes the matrix as xgboost wants examples as rows
        #each input category is a column
        x_t_data = x_data.T
        #converts to numpy array
        x_data = x_t_data.to_numpy()
        #removes the first row as its a leftover label
        x_data = np.delete(x_data,0,0)
        x_data = pd.DataFrame(data=x_data)
        categorical_data = x_data[[1,2,10,110,210,310,410,510,610,710,810,910]]
        x_data = x_data.drop([1,2,10,110,210,310,410,510,610,710,810,910], axis = 1)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe = ohe.fit(categorical_data)
        categorical_data = ohe.transform(categorical_data)
        categorical_data = pd.DataFrame(categorical_data)
        x_data = pd.concat([x_data, categorical_data], axis = 1)
        x_data = x_data.to_numpy()
    else:
        #transposes the matrix as xgboost wants examples as rows
        #each input category is a column
        x_t_data = x_data.T
        #converts to numpy array
        x_data = x_t_data.to_numpy()
        #removes the first row as its a leftover label
        x_data = pd.DataFrame(x_data)
        categorical_data = x_data[[1,2,10,110,210,310,410,510,610,710,810,910]]
        #print(categorical_data)
        x_data = x_data.drop([1,2,10,110,210,310,410,510,610,710,810,910], axis = 1)
        categorical_data = enc.transform(categorical_data)
        #print(categorical_data)
        categorical_data = pd.DataFrame(categorical_data)
        x_data = pd.concat([x_data, categorical_data], axis = 1)
        x_data = x_data.to_numpy()
        #print(x_data.shape)
        ohe = enc
    return x_data, ohe

#flag = 0 (DNN)
#flag = 1 (CNN)
def eval_dl(x,y,k,flag):
    cv = StratifiedKFold(n_splits=k,shuffle=True)
    best_model = []
    results = []
    highest = 0
    i = 1
    for train,test in cv.split(x,y):
        if(flag == 0):
            model = create_DNN(x[train].shape[1])
        if(flag == 1):
            x = x.reshape(x.shape[0], x.shape[1], 1)
            model = build_CNN(x[train].shape[1])
        bs = ((x[train].shape[0])/20)
        bs = round(bs)
        history = model.fit(x[train], y[train], validation_data=(x[test], y[test]), epochs = 200, batch_size=bs)
        _, accuracy = model.evaluate(x[test], y[test], batch_size=bs, verbose=0)
        accuracy = accuracy * 100
        print("accuracy for model " + str(i) + " is " + str(accuracy))
        if(accuracy > highest):
            highest = accuracy
            best_model = model
        results.append(accuracy)
        i = i + 1
    hx = ("highest accuracy is: " + str(highest))
    hav = ("Training Testing Accuracy: %.2f%% (%.2f%%)" % (np.mean(results), np.std(results)))
    return best_model, hx, hav

def main():
    best_xgb = pickle.load(open("best_accuracy_xgb.dat", "rb"))
    f = load_features(best_xgb)
    print(f)
    
    x_data = pd.read_csv('assembled_stat_matrix.csv')
    na_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    x_data, ohe = ohe_data(x_data, na_enc, 0)
    print(x_data.shape)
    x_data = x_data[:,f]
    print(x_data)
    print(x_data.shape)

    #loads the ylabel matrix,
    y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
    #transposes y_label
    y_t_label = y_label.T
    #converts to numpy
    y_label = y_t_label.to_numpy()
    #removes the first row, as its not an accruate outcome label, its just a row label
    y_label = np.delete(y_label, 0, 0)
    print(y_label.shape)

    bm, dnn_high, dnn_av = eval_dl(x_data, y_label, 10, 0)
    c_bm, cnn_high, cnn_av = eval_dl(x_data, y_label, 10, 1)
    print(dnn_high)
    print(dnn_av)
    print(cnn_high)
    print(cnn_av)


if __name__ == '__main__':
    main()
