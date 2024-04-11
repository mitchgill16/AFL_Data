#will run the previous rounds predictions to make sure
#PAV tables are up to date with correct lineups
#eg. in case Chris Scott pulled a late change after predictions were run.
# arg statements

#import packages
import xgboost as xgb
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
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score, KFold
import re
import math
from matplotlib import pyplot as plt
import subprocess
import sys
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from numpy import sort
import warnings
warnings.filterwarnings('ignore')
#get headers
#feed this into a bigger function which specifies the amount of games to go through
def get_headers(n_games):
    headers = ['Round', 'Home_Team', 'Away_Team', 'Venue', 'H_PAV_Sum', 'A_PAV_Sum']
    example_file = pd.read_csv('Data/Fremantle_clean_stats.csv')
    cl_h = example_file.columns
    cl_h = cl_h[:-5]
    ladder_header = ['Ladder Pos_H', 'Form_H', 'Season Wins_H', 'Season Loss_H', 'Season Draw_H']
    headers = [*headers, *ladder_header]
    j = 1
    while j <= n_games:
        for x in cl_h:
            if 'Match_ID' in x or 'Year' in x:
                continue
            x = 'H_'+ x + ' n-' + str(j)
            headers.append(x)
        j = j + 1
    j = 1
    ladder_header = ['Ladder Pos_A', 'Form_A', 'Season Wins_A', 'Season Loss_A', 'Season Draw_A']
    headers = [*headers, *ladder_header]
    while j <= n_games:
        for x in cl_h:
            if 'Match_ID' in x or 'Year' in x:
                continue
            x = 'A_'+ x + ' n-' + str(j)
            headers.append(x)
        j = j + 1
    return headers

def clean_headers(h):
    headers = []
    for x in h:
        if '<' in x or '>' in x:
            x = x.replace('<',"lt_")
            x = x.replace('>', "gt_")
            #print(x)
        headers.append(str(x))
    return headers

def generate_categorical_headers(h):
    cat_var = ['Round', 'Home_Team', 'Away_Team']
    skip = 0
    for x in h:
        if 'Round' in x:
            if (skip == 0):
                skip = 1
                continue
            cat_var.append(x)
            #print(x)
        elif 'Team_against_ID' in x:
            #print(x)
            cat_var.append(x)
        elif 'Venue' in x:
            cat_var.append(x)
    return cat_var

#one hot encode data and transform the X_data
#first redo, find the categorial variables
def ohe_data(x_data, enc, flag,cat_var):
    #data has not been previously one hot encoded
    if (flag == 0):
        #get columns with categorical data and drop from main DF
        categorical_data = x_data[cat_var]
        x_data = x_data.drop(cat_var, axis = 1)
        #define and fit new OHE. Use it on our categorical data by transforming
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe = ohe.fit(categorical_data)
        categorical_data = ohe.transform(categorical_data)
        #get feature names for better labelling
        fn = ohe.get_feature_names(cat_var)
        #make a dataframe with new OHE data and feature names
            #would have been good to have coded it like this for my Masters project...
        categorical_data = pd.DataFrame(categorical_data)
        categorical_data.columns = fn
        #ensure that it won't get cranky about any different indexes(shouldn't be any but just a good check)
        x_data.reset_index(drop=True, inplace=True)
        categorical_data.reset_index(drop=True, inplace=True)
        #concatenate along column axis
        x_data = pd.concat([x_data, categorical_data], axis = 1)
    else:
        #same as above except used already fitted ohe
        categorical_data = x_data[cat_var]
        x_data = x_data.drop(cat_var, axis = 1)
        categorical_data = enc.transform(categorical_data)
        fn = enc.get_feature_names(cat_var)
        categorical_data = pd.DataFrame(categorical_data)
        categorical_data.columns = fn
        x_data.reset_index(drop=True, inplace=True)
        categorical_data.reset_index(drop=True, inplace=True)
        x_data = pd.concat([x_data, categorical_data], axis = 1)
        ohe = enc
    return x_data, ohe

def load_models(n):
    clf = pickle.load(open("Models/best_xgb_clas_no2020_"+str(n)+"_games.dat", "rb"))
    reg = pickle.load(open("Models/best_xgb_reg_no2020_"+str(n)+"_games.dat", "rb"))
    ohe = pickle.load(open("Models/ohe_"+str(n)+"_no_2021_games.dat", "rb"))
    return clf, reg, ohe

def create_prev_games(team_id, n_games, teams):
    ma = None
    current_team = (teams[str(team_id)])
    team_string = current_team+"_clean_stats.csv"
    df = pd.read_csv("Data/"+team_string)
    df = df.iloc[::-1]
    df = df.head(n_games)
    df = df.reset_index()
    #print(df)
    #drops ladder stats
    #finds where in the dataframe the current match is
    #splits dataframe into game data and end of round ladder data
    l_df = df.iloc[:,-5:]
    t_df = df.iloc[: , :-5]
    #turns the WWWLL form column into # of W
    n_form = []
    for x in l_df['form']:
        if(len(x)<n_games):
            y=float(x.count("W"))
            n_form.append(y)
        else:
            x=x[-n_games:]
            y=float(x.count("W"))
            n_form.append(y)
    l_df['form'] = n_form

    #checks to make sure there is enough games to go through
    #start match array with the ladder values from end of previous round (as this would be current for predicting round)
    ma = l_df.loc[0].values
    #finds both labels for models
    #start from the previous game to current game
    #i is to know how many games included
    i = 0
    #j finds the previous game and allows for 2020 exclusion
    while i < n_games:
        cg = t_df.loc[i][3:].values
        ma = [*ma, *cg]
        i = i + 1
    return ma

def combine_prev_games(home_id, away_id, round_num, venue, h_pav, a_pav, home_array, away_array):
    current_example_array = [round_num, home_id, away_id, venue, h_pav, a_pav]
    current_example_array.extend(home_array)
    current_example_array.extend(away_array)
    return current_example_array

def predict(home_id, away_id, venue, round_num, h_pav, a_pav, n, teams):

    cea_df = []
    home_array = create_prev_games(home_id, n, teams)
    away_array = create_prev_games(away_id, n, teams)
    cea = combine_prev_games(home_id, away_id, round_num, venue, h_pav, a_pav, home_array, away_array)

    cea_df.append(cea)
    cea_df = pd.DataFrame(cea_df)
    h = get_headers(n)
    cea_df.columns = h

    clf, reg, ohe = load_models(n)
    h = get_headers(n)
    h = clean_headers(h)
    cv = generate_categorical_headers(h)
    x_data, ohe = ohe_data(cea_df, ohe, 1, cv)
    #I don't think this does anything, but I'm too scared to move it
    feature_names = x_data.columns

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    x_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_data.columns.values]

    #feature select the data and load in the feature selected model
    fs_filename = 'Models/fs_criteria_'+str(n)+'.dat'
    selector = pickle.load(open(fs_filename, "rb"))
    selected_x_data = selector.transform(x_data)
    selected_x_data = pd.DataFrame(selected_x_data)
    fs_clf = pickle.load(open("Models/best_xgb_clas_FS_no2020_"+str(n)+"_games.dat", "rb"))

    #check selection applied
    print("Performing Feature Selection")
    print(x_data.shape)
    print(selected_x_data.shape)

    y = fs_clf.predict(selected_x_data)
    yp = fs_clf.predict_proba(selected_x_data)
    my = reg.predict(x_data)
    my[0] = abs(my[0])
    my[0] = round(my[0],0)
    if(y < 0.5):
        p = yp[:,0]*100
        p = str(p)
        #Could somehow make this print statement into a javascript thing for django?
        #print(teams[str(home_id)] + "(HOME) is predicted to win against "+teams[str(away_id)]+" with a "+p[1:-1]+"% chance by " + str(my[0]) +" points")
    elif(y > 0.5):
        p = yp[:,1]*100
        p = str(p)
        #print(teams[str(away_id)] + "(AWAY) is predicted to win against "+teams[str(home_id)]+"  with a "+p[1:-1]+"% chance by " + str(my[0]) +" points")
    else:
        print("DRAW")
    return y, my[0]

def calc_sum_pav(year, rnd, team_int):
    #do calc
    g = gad()
    team_dict = g.createTeamDict()
    r_dict = create_R_TeamDict()
    current_team = (team_dict[str(team_int)])
    df = pd.read_csv("Data/"+current_team+'_clean_stats.csv')
    current_r_team = (r_dict[str(team_int)])
    lineups = pd.read_csv("R_Code/all_lineups.csv")
    #lower_case the line ups
    lineups['player.playerName.givenName'] = lineups['player.playerName.givenName'].str.lower()
    lineups['player.playerName.surname'] = lineups['player.playerName.surname'].str.lower()
    #filter line ups to the current round year and team
    lineups = lineups[lineups.isin([current_r_team]).any(axis=1)]
    lineups = lineups[lineups.isin([year]).any(axis=1)]
    lineups = lineups[lineups.isin([rnd]).any(axis=1)]
    lineups['team'] = team_int
    reduced = False
    #still waiting on PAVS
    if(rnd < 3):
        lineups['year'] = (year-1)
        reduced = True
    lineups.columns = ['year', 'teamname', 'roundNumber', 'firstname', 'surname', 'team']
    cols = ['team', 'year', 'firstname', 'surname']
    lineups = lineups[cols]
    all_pavs = pd.read_csv("R_Code/all_player_PAVs.csv")
    #lowcase the all_pavs
    all_pavs['firstname'] = all_pavs['firstname'].str.lower()
    all_pavs['surname'] = all_pavs['surname'].str.lower()

    #name edits
    lineups.firstname = lineups.firstname.str.split(' ').str[0]
    all_pavs.firstname = all_pavs.firstname.str.replace(' ','')

    #funky code to get players that do not exist in the 2022 PAVs
    xyz = lineups
    xyz = (
    xyz.merge(all_pavs,
              on=['team', 'year', 'firstname', 'surname'],
              how='outer',
              indicator=True)
    .query('_merge != "both"')
    .drop(columns='_merge'))
    xyz = xyz[xyz['PAV_total'].isna()]

    #drop the PAV_total column
    xyz = xyz.drop(['PAV_total', 'team'], axis = 1)
    #so we don't subtract too many years
    if(not reduced):
        xyz['year'] = xyz['year'] - 1
    #merge with all_pavs but the year before incase they've previously had a PAV
    xyz = xyz.merge(all_pavs, how='inner', on=['year', 'firstname', 'surname'])


    lineups = lineups.merge(all_pavs, how='inner', on=['team', 'year', 'firstname', 'surname'])
    #generate an old pav to add
    old_pav = 0
    if(xyz.shape[0] > 0):
        old_pav = xyz['PAV_total'].sum()
    print("pav from prev year = "+str(old_pav))
    ### if the line up exists
    if(lineups.shape[0] > 0):
        pav = lineups['PAV_total'].sum()
    #doesn't exist and should be obvious to do soething here
    else:
        pav = 999
    print(pav)
    #round pav because it get weird about it
    pav = round(pav,2)
    old_pav = round(old_pav, 2)
    pav = pav + old_pav
    print(pav)
    pav_array = [year, rnd, team_int, pav]
    return pav_array

def create_venue_alias_dict():
    vDict = {
        "Heritage Bank Stadium" : "Metricon Stadium",
        "UTAS Stadium" : "University of Tasmania Stadium",
        "AAMI Stadium" : "Adelaide Oval",
        "Domain Statium" : "Optus Stadium",
        "ENGIE Stadium" : "GIANTS Stadium",
        "People First Stadium" : "Metricon Stadium",
        "TIO Traeger Park" : "TIO Stadium"
        }
    return vDict

def create_R_TeamDict():
    teams = {
    "1" : "Adelaide Crows",
    "2" : "Brisbane Lions",
    "3" : "Carlton",
    "4" : "Collingwood",
    "5" : "Essendon",
    "6" : "Fremantle",
    "7" : "Geelong Cats",
    "8" : "Gold Coast Suns",
    "9" : "GWS Giants",
    "10": "Hawthorn",
    "11": "Melbourne",
    "12": "North Melbourne",
    "13": "Port Adelaide",
    "14": "Richmond",
    "15": "St Kilda",
    "16": "Sydney Swans",
    "17": "West Coast Eagles",
    "18": "Western Bulldogs"
    }
    return teams

#function to get the PAV for a team in round and year
def get_pav(season, round_num, team_id):
    p_df = pd.read_csv('R_Code/all_team_pavs.csv')
    test_pav = p_df.loc[(p_df['Year'] == season) & (p_df['Round'] == round_num) & (p_df['Team_ID'] == team_id)]
    x = test_pav['Player_PAV_Total'].values[0]
    y = test_pav['Player_PAV_Total'].values
    print(y)
    return x

def main():
    season = int(sys.argv[1])
    round_num= int(sys.argv[2])
    rdict = create_R_TeamDict()
    venue_df = pd.read_csv("R_Code/all_venues.csv")
    venue_df = venue_df.loc[(venue_df['round.year'] == season) & (venue_df['round.roundNumber'] == round_num)]
    home_teams = venue_df['match.homeTeam.name'].values
    away_teams = venue_df['match.awayTeam.name'].values
    home_team_ids = []
    for x in home_teams:
        team_id = list(rdict.values()).index(x) + 1
        home_team_ids.append(team_id)
    away_team_ids = []
    for x in away_teams:
        team_id = list(rdict.values()).index(x) + 1
        away_team_ids.append(team_id)
    home_teams = home_team_ids
    away_teams = away_team_ids
    venues = venue_df['venue.name'].values
    vdict = create_venue_alias_dict()
    v_int = 0
    while v_int < len(venues):
        current_ven = venues[v_int]
        if(current_ven in vdict):
            ven_alias = vdict.get(current_ven)
            venues[v_int] = ven_alias
        v_int = v_int + 1
    print(home_team_ids)
    print(away_team_ids)
    print(venues)
    start_match = 0
    end_match = len(venues)

    ###################
    ##### RUN :) ######
    ###################


    #load in dictionaries
    g = gad()
    teams = g.createTeamDict()

    df = pd.read_csv("R_Code/all_lineups.csv")
    print(df.shape)
    to_drop = df.index[(df['year'] == season) & (df['round.roundNumber'] == round_num) ]
    df = df.drop(to_drop)
    print(df.shape)
    df.to_csv("R_Code/all_lineups.csv", header=True, index=False)

    #update the all_lineups.csv
    subprocess.call(["/usr/bin/Rscript", "R_Code/update_lineups.R", str(season), str(round_num)])

    #check shape slightly reduces if lineup had previously existed
    df = pd.read_csv("R_Code/all_lineups.csv")
    print(df.shape)

    #uses the now updated all_lineups.csv to calculate PAVs for the specified games
    #updates the all_team_pavs file for easier access below and if retraining models
    #maybe chuck a cheeky remove duplicates and sort by year round in here for error checks
    pa = []
    for x in range(start_match, end_match):
        home_pav_array = calc_sum_pav(season, round_num, home_teams[x])
        pa.append(home_pav_array)
        away_pav_array = calc_sum_pav(season, round_num, away_teams[x])
        pa.append(away_pav_array)
    pav_df = pd.DataFrame(pa, columns=['Year', 'Round', 'Team_ID', 'Player_PAV_Total'])
    print(pav_df)

    #drop previous entries to all_team_pavs in the round and year
    all_pav_df = pd.read_csv('R_Code/all_team_pavs.csv')
    print(all_pav_df.shape)
    pav_to_drop = all_pav_df.index[(all_pav_df['Year'] == season) & (all_pav_df['Round'] == round_num) ]
    all_pav_df = all_pav_df.drop(pav_to_drop)
    print(all_pav_df.shape)

    all_pav_df = pd.concat([all_pav_df, pav_df], ignore_index=True)

    #remove duplicates and sort by year then round incase of multiple runnings or stupidity
    print(all_pav_df.shape)
    all_pav_df = all_pav_df.drop_duplicates()
    print(all_pav_df.shape)
    all_pav_df = all_pav_df.sort_values(["Year", "Round"], ascending = (True, True))
    all_pav_df.to_csv("R_Code/all_team_pavs.csv", header=True, index=False)

    tip_array = []
    margin_tip_array = []
    predict_round_num = round_num

    #Run the predictions
    # n is which n_games model
    n = 2

    i=start_match
    while i<end_match:
        home_id = home_teams[i]
        away_id = away_teams[i]

        home_pav = get_pav(season, round_num, home_id)
        away_pav = get_pav(season, round_num, away_id)

        venue = venues[i]
        tip, margin_tip = predict(home_id, away_id, venue, predict_round_num, home_pav, away_pav, n, teams)
        print("n = 2")
        print(tip)
        tip_array.append(tip[0])
        margin_tip_array.append(margin_tip)
        i = i + 1

    i = start_match
    n = 10
    while i<end_match:
        home_id = home_teams[i]
        away_id = away_teams[i]

        home_pav = get_pav(season, round_num, home_id)
        away_pav = get_pav(season, round_num, away_id)

        venue = venues[i]
        tip, margin_tip = predict(home_id, away_id, venue, predict_round_num, home_pav, away_pav, n, teams)
        print("n = 10")
        print(tip)
        tip_array.append(tip[0])
        margin_tip_array.append(margin_tip)
        i = i + 1

    i = start_match
    n = 3
    while i<end_match:
        home_id = home_teams[i]
        away_id = away_teams[i]

        home_pav = get_pav(season, round_num, home_id)
        away_pav = get_pav(season, round_num, away_id)

        venue = venues[i]
        tip, margin_tip = predict(home_id, away_id, venue, predict_round_num, home_pav, away_pav, n, teams)
        print("n = 3")
        print(tip)
        tip_array.append(tip[0])
        margin_tip_array.append(margin_tip)
        i = i + 1

    l = len(tip_array)
    tip_games = int(l/3)
    tip_df = pd.DataFrame({'n=2': tip_array[0:1*tip_games],
     'n=3':tip_array[2*tip_games:3*tip_games],
     'n=10':tip_array[1*tip_games:2*tip_games]})
    tip_df['mean'] = tip_df.mean(axis=1)
    print(tip_df)
    margin_df = pd.DataFrame({'n=2': margin_tip_array[0:1*tip_games],
     'n=3':margin_tip_array[2*tip_games:3*tip_games],
     'n=10':margin_tip_array[1*tip_games:2*tip_games]})
    margin_df['mean'] = margin_df.mean(axis=1)

    i=start_match
    while i<end_match:
        home_id = home_teams[i]
        away_id = away_teams[i]
        winner = (tip_df['mean'][i])
        winner_margin = margin_df['mean'][i]
        i = i + 1
        if(winner < 0.5):
            print(teams[str(home_id)] + "(HOME) is predicted to win against "+teams[str(away_id)]+" by " + str(winner_margin) +" points")
        elif(winner > 0.5):
            print(teams[str(away_id)] + "(AWAY) is predicted to win against "+teams[str(home_id)]+" by " + str(winner_margin) +" points")
        else:
            print("DRAW")

if __name__ == '__main__':
    main()
