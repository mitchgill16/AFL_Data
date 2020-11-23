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

def predict(model,home_id, away_id, round, teams, pda):
    home_array = create_5_most_recent(home_id, teams)
    away_array = create_5_most_recent(away_id, teams)
    current_example_array = combine_prev5(home_id, away_id, round, home_array, away_array)
    X = np.array(current_example_array)
    X = np.reshape(X,(1,len(X)))
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

def param_search(x_data):

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


def main():
    g = gad()
    teams = g.createTeamDict()
    #makes an array that keeps track of how many wins a team has for the random run
    pda = np.zeros(shape=19)
    i = 0
    #loads the input data from the assembled matrix in assemble_df.py
    x_data = pd.read_csv('assembled_stat_matrix.csv')
    gm = 1
    rl = 1
    md = 8
    lr =0.2
    ts = 0.2
    mcw = 1
    model = param_search(x_data)

    while(i<100):
        print("in the "+str(i)+" loop")
        seed = randint(0,5000)
        model, accuracy = check_xb_model(model, seed, x_data)
        if(accuracy < 68.0):
            print("not accurate enough")
            continue
        #predict upcoming games
        #if there is no 'home games' due to covid, do the reverse home/away structure for each game
        #the pda array should keep a track who wins for each seed, to hopefully minimise randomness
        predict(model,13, 14, 26, teams, pda)
        predict(model,14, 13, 26, teams, pda)

        predict(model,2, 7, 26, teams, pda)
        predict(model,7, 2, 26, teams, pda)


        i = i+1
    print(pda)
    determine_winner(13, 14, pda, teams)
    determine_winner(2, 7, pda, teams)

if __name__ == '__main__':
    main()
