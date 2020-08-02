import xgboost as xgb
#import torch.nn as nn
#import touch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint
from Gather_AFL_Data import gatherer as gad

def create_xb_model(t, l, m, r, g, mcw, seed, x_data):
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
    test_size = t
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=test_size, random_state=seed)
    print(X_test.shape)
    print("seed is " + str(seed))
    #says the model is XGBclassfier which means binary data
    model = xgb.XGBClassifier(learning_rate=l, max_depth=m, reg_lambda=r, gamma=g, min_child_weight=mcw)
    #trains the model, and makes the y shape as (m,) instead of (m,1)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    #sees how accurate the model was when testing the test set
    accuracy = accuracy_score(y_test, predictions)
    pcent = accuracy * 100.0
    print("The accuracy of this model is" + str(pcent))
    return model, pcent


def find_best_model():
    accuracy_matrix = np.loadtxt("optimise.csv", delimiter=",")
    average_columns = np.mean(accuracy_matrix, axis = 0)
    best_av = 0
    jj = 0
    ii = 0
    print(len(average_columns))
    for av in average_columns:
        print(av)
        if(ii>178):
            if(av>best_av):
                best_av = av
                jj = ii
        ii = ii+1
    print("with a jj value of "+ str(jj))
    print("with an average of " + str(best_av))

def optimse():
    #loads the input data from the assembled matrix in assemble_df.py
    x_data = pd.read_csv('assembled_stat_matrix.csv')
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
    print(x_data.shape)
    print(y_label.shape)
    #splits up the X and Y data into training and test
    #gm = 2, rl = 2, md = 6, lr =0.2, ts = 0.1
    seed = 28
    total = 0
    i = 0
    j = 0
    highest = 0
    best = 0
    best_j = 9999999
    ts = [0.2, 0.25, 0.3]
    lr = [0.3, 0.4]
    md = [6, 7 ,8]
    rl = [0, 1]
    gm = [0, 1]
    models = []
    accuracy_matrix = np.zeros(shape=(10,72))
    #average out of 10 random seeds
    while(i<10):
        k = 0
        #random int generation
        seed = randint(0,1000)
        #average out of 6 test/training splits
        for t in ts:
            test_size = t
            X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=test_size, random_state=seed)
            print(X_test.shape)
            #says the model is XGBclassfier which means binary data
            #goes through each of parameters I'm testing
            for l in lr:
                for m in md:
                    for r in rl:
                        for g in gm:
                            model = xgb.XGBClassifier(learning_rate=l, max_depth=m, reg_lambda=r, gamma=g)
                            #trains the model, and makes the y shape as (m,) instead of (m,1)
                            model.fit(X_train, y_train.ravel())
                            #uses unseen data to predict
                            y_pred = model.predict(X_test)
                            predictions = [round(value) for value in y_pred]
                            #sees how accurate the model was when testing the test set
                            accuracy = accuracy_score(y_test, predictions)
                            pcent = accuracy * 100.0
                            print("The accuracy of this model" +str(i)+":"+str(k)+" is" + str(pcent))
                            #appends the model paramters to the model array
                            models.append(model)
                            #appends what the test ratio split was
                            #puts the accuracy into its appropriate row and column
                            #row is the current random seed example
                            #column is the current combination of parameters
                            #will allow for an average across 10 seeds per combination of parameter
                            accuracy_matrix[i][k] = pcent
                            #checks what the best was
                            if (pcent > highest):
                                highest = pcent
                                best = seed
                                best_j = j
                            j = j + 1
                            k = k + 1
        i = i + 1
    np.savetxt("optimise_2.csv", accuracy_matrix, delimiter=",")
    np.savetxt("models.csv", models, delimeter=",")
    average_columns = np.mean(accuracy_matrix, axis = 0)
    best_av = 0
    jj = 0
    ii = 0
    for av in average_columns:
        print(av)
        if(av>best_av):
            best_av = av
            jj = ii
        ii = ii+1
    print("with a jj value of "+ str(jj))
    print("with an average of " + str(best_av))

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

# def make_dl_model():
#     #loads the input data from the assembled matrix in assemble_df.py
#     x_data = pd.read_csv('assembled_stat_matrix.csv')
#     #transposes the matrix as xgboost wants examples as rows
#     #each input category is a column
#     x_t_data = x_data.T
#     #converts to numpy array
#     x_data = x_t_data.to_numpy()
#     #removes the first row as its a leftover label
#     x_data = np.delete(x_data,0,0)
#     #converts np to tensor
#     x_tensor = torch.from_numpy(x_data)
#     #loads the ylabel matrix,
#     y_label = pd.read_csv('assembled_labelled_ymatrix.csv')
#     #transposes y_label
#     y_t_label = y_label.T
#     #converts to numpy
#     y_label = y_t_label.to_numpy()
#     #removes the first row, as its not an accruate outcome label, its just a row label
#     y_label = np.delete(y_label, 0, 0)
#     #coverts np to tensor
#     y_tensor = torch.from_numpy(y_label)


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

    while(i<100):
        print("in the "+str(i)+" loop")
        seed = randint(0,5000)
        model, accuracy = create_xb_model(ts, lr, md, rl, gm, mcw, seed, x_data)
        if(accuracy < 66.0):
            print("not accurate enough")
            continue
        #predict upcoming games
        #if there is no 'home games' due to covid, do the reverse home/away structure for each game
        #the pda array should keep a track who wins for each seed, to hopefully minimise randomness
        predict(model,18, 14, 9, teams, pda)
        predict(model,14, 18, 9, teams, pda)

        predict(model,11, 13, 9, teams, pda)
        predict(model,13, 11, 9, teams, pda)

        predict(model,3, 10, 9, teams, pda)
        predict(model,10, 3, 9, teams, pda)

        predict(model,5, 2, 9, teams, pda)
        predict(model,2, 5, 9, teams, pda)

        predict(model,12, 1, 9, teams, pda)
        predict(model,1, 12, 9, teams, pda)

        predict(model,15, 16, 9, teams, pda)
        predict(model,16, 15, 9, teams, pda)

        predict(model,17, 7, 9, teams, pda)
        predict(model,7, 17, 9, teams, pda)

        predict(model,8, 9, 9, teams, pda)
        predict(model,9, 8, 9, teams, pda)

        predict(model,6, 4, 9, teams, pda)
        predict(model,4, 6, 9, teams, pda)


        i = i+1
    print(pda)
    determine_winner(18, 14, pda, teams)
    determine_winner(11, 13, pda, teams)
    determine_winner(3, 10, pda, teams)
    determine_winner(5, 2, pda, teams)
    determine_winner(12, 1, pda, teams)
    determine_winner(15, 16, pda, teams)
    determine_winner(17, 7, pda, teams)
    determine_winner(8, 9, pda, teams)
    determine_winner(6, 4, pda, teams)

if __name__ == '__main__':
    main()
