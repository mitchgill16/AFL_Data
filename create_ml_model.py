import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint
from Gather_AFL_Data import gatherer as gad

def create_xb_model():
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
    seed = 28
    total = 0
    i = 10
    highest = 0
    best = 0
    while(i<11):
        #seed = randint(0,50)
        seed = 18
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, test_size=test_size, random_state=seed)
        print(X_test.shape)
        #says the model is XGBclassfier which i think is binary??
        model = xgb.XGBClassifier(learning_rate=0.2, max_depth=6, reg_lambda=1, gamma=0)
        #trains the model, and makes the y shape as (m,) instead of (m,1)
        model.fit(X_train, y_train.ravel())
        #uses unseen data to predict
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        for value in y_pred:
            print(value)
        accuracy = accuracy_score(y_test, predictions)
        pcent = accuracy * 100.0
        total = pcent + total
        print("The accuracy of this model is" + str(pcent))
        print(model)
        i = i + 1
        if (pcent > highest):
            highest = pcent
            best = seed
    average = total/10
    print("Average is " + str(average))
    print("Best seed is " + str(seed))
    return model


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

def predict(model,home_id, away_id, round, teams):
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
    elif(y > 0.5):
        p = (y-0.5)*2
        print(teams[str(away_id)] + " is predicted to win")
    else:
        print("DRAW")


def main():
    g = gad()
    teams = g.createTeamDict()
    model = create_xb_model()
    #create prev 5 for upcoming games.
    predict(model,1, 2, 5, teams)
    predict(model,1, 3, 5, teams)
    predict(model,1, 4, 5, teams)
    predict(model,1, 5, 5, teams)
    predict(model,1, 6, 5, teams)
    predict(model,1, 7, 5, teams)
    predict(model,1, 8, 5, teams)
    predict(model,1, 9, 5, teams)
    predict(model,1, 10, 5, teams)
if __name__ == '__main__':
    main()
