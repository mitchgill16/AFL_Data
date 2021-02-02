import xgboost as xgb
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

# def create_xb_model(t, l, m, r, g, mcw, seed, x_data):
#     #transposes the matrix as xgboost wants examples as rows
#     #each input category is a column
#     x_t_data = x_data.T
#     #converts to numpy array
#     x_data = x_t_data.to_numpy()
#     #removes the first row as its a leftover label
#     x_data = np.delete(x_data,0,0)
#     #loads the ylabel matrix,
#     y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
#     #transposes y_label
#     y_t_label = y_label.T
#     #converts to numpy
#     y_label = y_t_label.to_numpy()
#     #removes the first row, as its not an accruate outcome label, its just a row label
#     y_label = np.delete(y_label, 0, 0)
#     test_size = t
#     X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=test_size, random_state=seed)
#     print(X_test.shape)
#     print("seed is " + str(seed))
#     #says the model is XGBclassfier which means binary data
#     model = xgb.XGBClassifier(learning_rate=l, max_depth=m, reg_lambda=r, gamma=g, min_child_weight=mcw)
#     #trains the model, and makes the y shape as (m,) instead of (m,1)
#     model.fit(X_train, y_train.ravel())
#     y_pred = model.predict(X_test)
#     predictions = [round(value) for value in y_pred]
#     #sees how accurate the model was when testing the test set
#     accuracy = accuracy_score(y_test, predictions)
#     pcent = accuracy * 100.0
#     print("The accuracy of this model is" + str(pcent))
#     return model, pcent

def check_xb_model(model, seed, x_data):
        #transposes the matrix as xgboost wants examples as rows
        #each input category is a column
        x_t_data = x_data.T
        #converts to numpy array
        x_data = x_t_data.to_numpy()
        #removes the first row as its a leftover label
        x_data = np.delete(x_data,0,0)
        #loads the ylabel matrix,
        y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
        #transposes y_label
        y_t_label = y_label.T
        #converts to numpy
        y_label = y_t_label.to_numpy()
        #removes the first row, as its not an accruate outcome label, its just a row label
        y_label = np.delete(y_label, 0, 0)
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.2, random_state=seed)
        #trains the model, and makes the y shape as (m,) instead of (m,1)
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        #sees how accurate the model was when testing the test set
        accuracy = accuracy_score(y_test, predictions)
        pcent = accuracy * 100.0
        print("The accuracy of this model is" + str(pcent))
        return model, pcent

def create_5_most_recent(team_id, teams):
    current_team = (teams[str(team_id)])
    team_string = current_team+"_stats.xlsx"
    t_df = pd.read_excel(team_string)
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

def predict(model,home_id, away_id, round, teams, pda, ohe):
    home_array = create_5_most_recent(home_id, teams)
    away_array = create_5_most_recent(away_id, teams)
    current_example_array = combine_prev5(home_id, away_id, round, home_array, away_array)
    cea = pd.DataFrame(current_example_array)
    X, na_enc = ohe_data(cea, ohe, 1)
    y = model.predict(X)
    print(str(y[0]))
    if(y < 0.5):
        p = (0.5-y)*2
        print(teams[str(home_id)] + " is predicted to win")
        pda[home_id] = pda[home_id] + 1
    elif(y > 0.5):
        p = (y-0.5)*2
        print(teams[str(away_id)] + " is predicted to win")
        pda[away_id] = pda[away_id] + 1
    else:
        print("DRAW")

def determine_winner(home_id, away_id, pda, teams):
    if(pda[home_id] > pda[away_id]):
        print(teams[str(home_id)] + " has been determined to win with a "+ str((pda[home_id]/(pda[home_id]+pda[away_id]))*100)+"% chance\n")
    elif(pda[away_id] > pda[home_id]):
        print(teams[str(away_id)] + " has been determined to win with a "+ str((pda[away_id]/(pda[home_id]+pda[away_id]))*100)+"% chance\n")
    else:
        print(teams[str(home_id)] + " + " + teams[str(away_id)] + " will draw!!!?!?!?\n")

def param_search(x_data, y_label):

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
    xgbclass = xgb.XGBClassifier(random_state=27022013)
    xgb_bayes_search = BayesSearchCV(xgbclass, space, n_iter=60, # specify how many iterations
                                    scoring=None, n_jobs=1, cv=5, verbose=3, random_state=42, n_points=12,
                                 refit=True)
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

def run_predictions(x, y, m, ohe, teams):
    pda = np.zeros(shape=19)
    i = 1
    results = []
    while(i<11):
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        for train,test in cv.split(x,y):
            prediction = m.fit(x[train],y[train].ravel()).predict_proba(x[test])
            print("variables for auroc curve done. Processing fold accuracy + checking best model")
            y_pred = m.predict(x[test])
            predictions = [round(value) for value in y_pred]
            #sees how accurate the model was when testing the test set
            accuracy = accuracy_score(y[test], predictions)
            pcent = accuracy * 100.0
            print("The accuracy of this model is" + str(pcent))
            results.append(pcent)

            predict(m,14, 3, 1, teams, pda, ohe)
            predict(m,3, 14, 1, teams, pda, ohe)
            predict(m,4, 18, 1, teams, pda, ohe)
            predict(m,18, 4, 1, teams, pda, ohe)
            predict(m,11, 6, 1, teams, pda, ohe)
            predict(m,6, 11, 1, teams, pda, ohe)
            predict(m,1, 7, 1, teams, pda, ohe)
            predict(m,7, 1, 1, teams, pda, ohe)
            predict(m,5, 10, 1, teams, pda, ohe)
            predict(m,10, 5, 1, teams, pda, ohe)
            predict(m,2, 16, 1, teams, pda, ohe)
            predict(m,16, 2, 1, teams, pda, ohe)
            predict(m,12, 13, 1, teams, pda, ohe)
            predict(m,13, 12, 1, teams, pda, ohe)
            predict(m,9, 15, 1, teams, pda, ohe)
            predict(m,15, 9, 1, teams, pda, ohe)
            predict(m,17, 8, 1, teams, pda, ohe)
            predict(m,8, 17, 1, teams, pda, ohe)

        i = i + 1
    print("Training Testing Accuracy: %.2f%% (%.2f%%)" % (np.mean(results), np.std(results)))
    return pda

    #do i <21
    #cv split thing shuffle True
    #predict
    #results
    #return pda

def main():
    g = gad()
    teams = g.createTeamDict()
    #makes an array that keeps track of how many wins a team has for the random run
    #loads the input data from the assembled matrix in assemble_df.py
    x_data = pd.read_csv('assembled_stat_matrix.csv')
    na_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    x_data, ohe = ohe_data(x_data, na_enc, 0)
    #loads the ylabel matrix,
    y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
    #transposes y_label
    y_t_label = y_label.T
    #converts to numpy
    y_label = y_t_label.to_numpy()
    #removes the first row, as its not an accruate outcome label, its just a row label
    y_label = np.delete(y_label, 0, 0)
    #model = param_search(x_data, y_label)
    model = pickle.load(open("xgb_model.dat", "rb"))
    pda = run_predictions(x_data, y_label, model, ohe, teams)
    print(pda)
    determine_winner(14, 3, pda, teams)
    determine_winner(4, 18, pda, teams)
    determine_winner(11, 6, pda, teams)
    determine_winner(1, 7, pda, teams)
    determine_winner(5, 10, pda, teams)
    determine_winner(2, 16, pda, teams)
    determine_winner(12, 13, pda, teams)
    determine_winner(9, 15, pda, teams)
    determine_winner(17, 8, pda, teams)

if __name__ == '__main__':
    main()
