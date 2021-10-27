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
#from fdnn import feature_extractor as fex
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


def create_n_most_recent(team_id, teams, n):
    current_team = (teams[str(team_id)])
    team_string = current_team+"_stats.xlsx"
    t_df = pd.read_excel("Data/"+team_string)
    col = list(t_df)
    #reversing allows us to find our current game and get the previous 5 in an easy way
    col.reverse()
    match_array = []
    j = 0
    #goes through the spreadsheet until it finds our match we want to look at
    #then sets j as 0 to allow the program to get the stats from 5 matches
    #adds it all to the match array
    for i in col:
        if(j>= 0 and j<n):
            y = 0
            for element in t_df[i]:
                #skips adding the year the game was played in to the data
                if(y == 0):
                    y = 1
                    continue
                match_array.append(element)
            #print(len(t_df[i]))
            j = j + 1
    return match_array

def combine_prev(home_id, away_id, round, home_array, away_array):
    current_example_array = [round, home_id, away_id]
    current_example_array.extend(home_array)
    current_example_array.extend(away_array)
    return current_example_array

##predict a given set of data
#model = classification win model generated in ML_models, saved as pickle
#mm = regression margin model generated in ML_models, saved as picle
#home_id = home team (1-18)
#away_id = away team (1-18)
#round = upcoming round #
#teams = team dict generated fromt Gather_AFL_Data.py
#ohe = one hot encoder object used to encode assemble_df to train models saved in ML_models
#venue = upcoming venue (will be one hot encoded in the same way)
#year = current year for the round
def predict(model, home_id, away_id, round, teams, ohe, mm, venue, year):
    #gets the 10 most recent games of data for home team
    home_array = create_n_most_recent(home_id, teams, 10)
    #add a function add things like upcoming venue, PAV of upcoming team & current ladder position
    #eg. h_PAV = find_PAV(home_id, round, year)
        #h_lad = find_ladder(home_id, round, year)

    away_array = create_n_most_recent(away_id, teams, 10)
    #add a function add things like upcoming venue, PAV of upcoming team & current ladder position

    #combines these two arrays to be the same inputs as the assemble_df.csv that the models are trained on
    current_example_array = combine_prev(home_id, away_id, round, home_array, away_array)
    cea = pd.DataFrame(current_example_array)

    #one hot encoded in the same way
    X, na_enc = ohe_data(cea, ohe, 1)

    #basic prediction structure
    #a win prediction and a margin prediction
    y = model.predict(X)
    my = mm.predict(X)
    if(y < 0.5):
        p = (0.5-y)*2
        #Could somehow make this print statement into a javascript thing for django?
        print(teams[str(home_id)] + "(HOME) is predicted to win by " + str(my[0]))
    elif(y > 0.5):
        p = (y-0.5)*2
        print(teams[str(away_id)] + "(AWAY) is predicted to win by " + str(my[0]))
    else:
        print("DRAW")

def ohe_data(x_data, enc):
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

def main():
    g = gad()
    teams = g.createTeamDict()
    #load from saved object when making model
    ohe = pickle.load("ohe_object.dat", "rb")
    win_model = pickle.load(open("xgb_model.dat", "rb"))
    margin_model = pickle.load(open("xgb_margin_model.dat", "rb"))

    #predict
    #// to do is to make some of this inputtable from command line depending how R scripts turn out
    i = 1
    while i<19:
        home_id=[i]
        away_id = [i+1]
        predict(win_model, home_id, away_id, round, teams, ohe, margin_model, venue, year)
        i = i + 2

if __name__ == '__main__':
    main()
