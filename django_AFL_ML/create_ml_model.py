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


def create_5_most_recent(team_id, teams):
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
        if(j>= 0 and j<5):
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

def combine_prev5(home_id, away_id, round, home_array, away_array):
    current_example_array = [round, home_id, away_id]
    current_example_array.extend(home_array)
    current_example_array.extend(away_array)
    return current_example_array

def predict(model,home_id, away_id, round, teams, pda, ohe, mm, mda, cda, cnn_flag):
    home_array = create_5_most_recent(home_id, teams)
    away_array = create_5_most_recent(away_id, teams)
    current_example_array = combine_prev5(home_id, away_id, round, home_array, away_array)
    cea = pd.DataFrame(current_example_array)
    X, na_enc = ohe_data(cea, ohe, 1)
    #delete out if you need
    if(cnn_flag == 1):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        my = mm.predict(X)
        if(my > 0):
            print(teams[str(home_id)] + "(HOME) is predicted to win by " + str(my[0]))
            pda[home_id] = pda[home_id] + 1
            mda[home_id] = mda[home_id] + my[0]
            cda[home_id] = cda[home_id] + 1
        elif(my < 0):
            my = -my
            print(teams[str(away_id)] + "(AWAY) is predicted to win by " + str(my[0]))
            pda[away_id] = pda[away_id] + 1
            mda[away_id] = mda[away_id] + my[0]
            cda[away_id] = cda[away_id] + 1
    else:
        y = model.predict(X)
        my = mm.predict(X)
        if(y < 0.5):
            p = (0.5-y)*2
            print(teams[str(home_id)] + "(HOME) is predicted to win by " + str(my[0]))
            pda[home_id] = pda[home_id] + 1
            if(my[0] > 0):
                mda[home_id] = mda[home_id] + my[0]
                cda[home_id] = cda[home_id] + 1
        elif(y > 0.5):
            p = (y-0.5)*2
            my = -my
            print(teams[str(away_id)] + "(AWAY) is predicted to win by " + str(my[0]))
            pda[away_id] = pda[away_id] + 1
            if(my[0] > 0):
                mda[away_id] = mda[away_id] + my[0]
                cda[away_id] = cda[away_id] + 1
        else:
            print("DRAW")

def determine_winner(home_id, away_id, pda, teams, mda, cda):
    if(pda[home_id] > pda[away_id]):
        print(teams[str(home_id)] + " has been determined to win with a "+ str((pda[home_id]/(pda[home_id]+pda[away_id]))*100)+"% chance")
        print("Average Margin Of: " +str((mda[home_id]/cda[home_id])))
    elif(pda[away_id] > pda[home_id]):
        print(teams[str(away_id)] + " has been determined to win with a "+ str((pda[away_id]/(pda[home_id]+pda[away_id]))*100)+"% chance")
        print("Average Margin Of: " + str((mda[away_id]/cda[away_id])))
    else:
        print(teams[str(home_id)] + " + " + teams[str(away_id)] + " will draw!!!?!?!?\n")
        print("determine by margin")
        print(teams[str(home_id)] + "Average Margin Of: " + str((mda[home_id]/cda[home_id])))
        print(teams[str(away_id)] + "Average Margin Of: " + str((mda[away_id]/cda[away_id])))

def param_search(x_data, y_label, class_reg):

    def on_step(optim_result):
        """
        Callback meant to view scores after
        each iteration while performing Bayesian
        Optimization in Skopt"""
        score = xgb_bayes_search.best_score_
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.2, random_state=27022013)
    space ={'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'min_child_weight': Integer(0, 10),
        'max_depth': Integer(0, 50),
        'max_delta_step': Integer(0, 20),
        'subsample': Real(0.01, 1.0, 'uniform'),
        'colsample_bytree': Real(0.01, 1.0, 'uniform'),
        'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
        'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
        'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
        'gamma': Real(1e-9, 0.5, 'log-uniform'),
        'min_child_weight': Integer(0, 5),
        'n_estimators': Integer(50, 200),
        'scale_pos_weight': Real(1e-6, 500, 'log-uniform')}
    if(class_reg == 0):
        xgbclass = xgb.XGBClassifier(random_state=27022013)
    else:
        xgbclass = xgb.XGBRegressor(random_state=27022013)
    xgb_bayes_search = BayesSearchCV(xgbclass, space, n_iter=60, # specify how many iterations
                                    scoring=None, n_jobs=1, cv=5, verbose=3, random_state=42, n_points=12,
                                 refit=True)
    kk = np.isinf(X_train)
    if True in kk:
    	print("aaaaaaa")
    kk = np.isinf(y_train)
    if True in kk:
    	print("reeeeeee")
    xgb_bayes_search.fit(X_train, y_train.ravel(), callback = on_step)
    print("BEST PARAMS ARE HERE")
    print(xgb_bayes_search.best_params_)
    model = xgb_bayes_search.best_estimator_

    return model

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

def run_predictions(x, y, m, my, mm, ohe, teams, games, g_round):
    #prediction of wins for each team
    pda = np.zeros(shape=19)
    #prediction of margin for each team
    mda = np.zeros(shape=19)
    #number of times margin agreed with who will win (pda)
    cda = np.zeros(shape=19)
    i = 1
    results = []
    error = []
    count = 0
    best_w = m
    high_w = 0
    best_m = mm
    high_m = 100
    cnn_check = 0
    while(i<11):
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        for train,test in cv.split(x,y):
            count = count + 1
            print("Model: " + str(count))
            prediction = m.fit(x[train],y[train].ravel()).predict_proba(x[test])
            margin_pred = mm.fit(x[train], my[train].ravel())
            print("variables for auroc curve done. Processing fold accuracy + checking best model")
            y_pred = m.predict(x[test])
            #print(y_pred)
            m_pred = mm.predict(x[test])
            #print(m_pred)
            predictions = [round(value) for value in y_pred]
            #sees how accurate the model was when testing the test set
            accuracy = accuracy_score(y[test], predictions)
            pcent = accuracy * 100.0
            print("The accuracy of this model is" + str(pcent))
            rmse = sqrt(mean_squared_error(m_pred, my[test]))
            print("The rmse of this model is" + str(rmse))
            results.append(pcent)
            error.append(rmse)
            #change the best model to equal current model
            if(pcent > high_w):
                print("found new best classify")
                best_w = m
                high_w = pcent
            if(rmse < high_m):
                print("found best new margin")
                best_m = mm
                high_m = rmse
            #the games being played in an array. Each pair of 2 is the teams playing against each other
            g = 0
            while (g<len(games)):
                predict(m,games[g], games[g+1], g_round, teams, pda, ohe, mm, mda,cda, 0)
                predict(m,games[g+1], games[g], g_round, teams, pda, ohe, mm, mda,cda, 0)
                g = g+2

        i = i + 1
    #Do one round of CNN prediction to add a bit of a different prediction for tight games
    x = x.reshape(x.shape[0], x.shape[1], 1)
    cv = KFold(n_splits=10, shuffle=True)
    for train,test in cv.split(x,y):
        cnn = build_CNN_model(x[train].shape[1])
        bs = ((x[train].shape[0])/20)
        bs = round(bs)
        history = cnn.fit(x[train], my[train], validation_data=(x[test], my[test]), epochs = 20, batch_size=bs)
        prediction = cnn.predict(x[test])
        print("variables for auroc curve done. Processing fold accuracy + checking best model")
        y_pred = prediction
        print(y_pred)
        all_preds = y_pred
        i = 0
        actual = my[test]
        predicted_wins = []
        actual_wins = []
        for omg in all_preds:
            if(omg >= 0):
                predicted_wins.append(0)
            else:
                predicted_wins.append(1)

            if(actual[i] >= 0):
                actual_wins.append(0)
            else:
                actual_wins.append(1)
            i = i + 1
        #sees how accurate the model was when testing the test set
        accuracy = accuracy_score(actual_wins, predicted_wins)
        pcent = accuracy * 100.0
        print("The accuracy of this model is" + str(pcent))
        results.append(pcent)
        rmse = sqrt(mean_squared_error(all_preds, my[test]))
        print("The rmse of this model is" + str(rmse))
        error.append(rmse)
        g = 0
        while (g<len(games)):
            predict(m,games[g], games[g+1], g_round, teams, pda, ohe, cnn, mda,cda, 1)
            predict(m,games[g+1], games[g], g_round, teams, pda, ohe, cnn, mda,cda, 1)
            g = g+2
    print("Training Testing Accuracy: %.2f%% (%.2f%%)" % (np.mean(results), np.std(results)))
    print("Training Testing Margins: %.2f%% (%.2f%%)" % (np.mean(error), np.std(error)))
    g = 0
    #determine winner from random forest of predictors method
    print("From random forest of predictor methods \n")
    while (g<len(games)):
        determine_winner(games[g], games[g+1], pda, teams, mda, cda)
        g = g + 2
    #determine winner from literally one prediction of the best models
    g = 0

    print("best accuracy is: " + str(high_w))
    print("best margin error is: " + str(high_m))
    print("from best model from all runs \n")
    while (g<len(games)):
        predict(best_w, games[g], games[g+1], g_round, teams, pda, ohe, best_m, mda, cda, 0)
        g = g + 2
    return pda, mda, best_w


def build_DNN_model(x_len):
    model = Sequential()
    model.add(Dense(63, input_dim = x_len))
    model.add(Activation('relu'))
    model.add(Dropout(0.03))
    model.add(BatchNormalization())

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.02))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(8))
    model.add(Activation('relu'))
    #add output layer
    model.add(Dense(1, activation='linear'))
    opt = tf.keras.optimizers.Adamax(learning_rate=0.003)

    model.compile(loss="mean_squared_error", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print(model.summary())
    return model

def build_CNN_model(x_len):
    #del model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=14,
                     input_shape=(x_len, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=16, kernel_size=10,
                     input_shape=(32, 1)))
    model.add(Activation('linear'))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=10, kernel_size=8,
                     input_shape=(16, 1)))
    model.add(Activation('linear'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='linear'))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(16, activation='linear'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    opt = tf.keras.optimizers.Adamax(learning_rate=0.003)#, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax"


    model.compile(loss="mean_squared_error", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    print(model.summary())
    return model

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
            model = build_DNN_model(x[train].shape[1])
        if(flag == 1):
            x = x.reshape(x.shape[0], x.shape[1], 1)
            model = build_CNN_model(x[train].shape[1])
        bs = ((x[train].shape[0])/20)
        bs = round(bs)
        history = model.fit(x[train], y[train], validation_data=(x[test], y[test]), epochs = 50, batch_size=bs)
        _, accuracy = model.evaluate(x[test], y[test], batch_size=bs, verbose=0)
        accuracy = accuracy * 100
        print("accuracy for model " + str(i) + " is " + str(accuracy))
        if(accuracy > highest):
            highest = accuracy
            best_model = model
        results.append(accuracy)
        i = i + 1
    print("highest accuracy is: " + str(highest))
    print("Training Testing Accuracy: %.2f%% (%.2f%%)" % (np.mean(results), np.std(results)))
    return best_model

def main():
    g = gad()
    teams = g.createTeamDict()
    #makes an array that keeps track of how many wins a team has for the random run
    #loads the input data from the assembled matrix in assemble_df.py
    x_data = pd.read_csv('Data/assembled_stat_matrix.csv')
    na_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    x_data, ohe = ohe_data(x_data, na_enc, 0)

    #loads the ylabel matrix,
    y_label = pd.read_csv('Data/assembled_labelled_ymatrix.csv')
    #transposes y_label
    y_t_label = y_label.T
    #converts to numpy
    y_label = y_t_label.to_numpy()
    #removes the first row, as its not an accruate outcome label, its just a row label
    y_label = np.delete(y_label, 0, 0)

    #loads margin as the y_label
    margin_label = pd.read_csv('Data/assembled_margin_ymatrix.csv')
    margin_t_label = margin_label.T
    margin_label = margin_t_label.to_numpy()
    margin_label = np.delete(margin_label, 0, 0)
    print(margin_label)


    #for predicting win
    model = param_search(x_data, y_label, 0)
    #for predicting margin
    margin_model = param_search(x_data, margin_label, 1)
    pickle.dump(model, open("xgb_model.dat", "wb"))
    pickle.dump(margin_model, open("xgb_margin_model.dat", "wb"))
    win_model = pickle.load(open("xgb_model.dat", "rb"))
    margin_model = pickle.load(open("xgb_margin_model.dat", "rb"))
    #dnn_model = build_DNN_model(x_data.shape[1])
    #best_model = predict_margin(x_data, margin_label, margin_model, ohe, teams)
    #best_model = predict_margin(x_data, margin_label, dnn_model, ohe, teams)

    # model = pickle.load(open("xgb_model.dat", "rb"))
    games = [6,7,14,2,9,16,8,18,11,10,15,13,12,5,4,3,1,17]
    round = 18
    pda, mda, best_xgb = run_predictions(x_data, y_label, win_model, margin_label, margin_model, ohe, teams, games, round)
    print(pda)
    print(mda)
    pickle.dump(best_xgb, open("best_accuracy_xgb.dat", "wb"))
    best_xgb = pickle.load(open("best_accuracy_xgb.dat", "rb"))

if __name__ == '__main__':
    main()
