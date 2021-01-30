#to do: run update with creation of new dataframes

#ideas for the future

##add ladder ranking for the each teams match
###  the ladder would have to have round 1 values in round 2, as its the ladder before the round 2 games
###  Round 1 games would be the position at the end of round 23 the previous year, finals stay the same as round 23
###  Round 1 2011, can be entered manually from 2010?

##Make a function that can find an average PAV for each game
### Have the players PAV from the previous year, so not sure how good it is?
### could try to find a way to see which players played per game, or just give
### the team an average PAV from the previous years?

#Takes the information created from Gather_AFL_Data.py and
#produces the input matrix (X) and the labelled array (Y)
#in a format that should be suitable for deep learning packages
#Matrix X is of NxM where N is number of rows.
#N should be each stat from the previous 5 games for each team playing
#M is the number of examples, so its every match from R7 2011 to current
#Array Y should be of 1xM, where each value is whether the home or away team Won
#Should be used to compute Y-hat for neural network

import numpy as np
import pandas as pd
import random as rand
import sys
from Gather_AFL_Data import gatherer as gad

def find_teams_playing(match_id, teams):
    i = 1
    match_id = str(match_id)
    team1 = 999
    team2 = 999
    while(i<19):
        current_team = (teams[str(i)])
        textfile = open(current_team+"_data.txt", 'r')
        lines = textfile.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            if(line == match_id):
                if(team1 == 999):
                    team1 = i
                else:
                    team2 = i
                break
        i = i+1
    return team1, team2

def determine_home_away(match_id, team1, team2, teams):
    current_team = (teams[str(team1)])
    team_string = current_team+"_stats.xlsx"
    #reads the team1 excel into a dataframe
    t1_df = pd.read_excel(team_string)
    #every column is now a list of elements.
    col = list(t1_df)
    #iterate through each column
    for i in col:
        #the MATCH_ID is the index as its the top row, good thinking previous Mitch
        if(i == match_id):
            #the 3rd element of the column (its 2 because arrays start at 0) is the H/A value
            ha_val = t1_df[i][2]
            round = t1_df[i][1]
            break
    #floats are fun
    if(ha_val == 0.0):
        home_id = team1
        away_id = team2
    elif(ha_val == 1.0):
        home_id = team2
        away_id = team1
    else:
        #somethings gone wrong
        home_id = 999
        away_id = 999
    return home_id, away_id, round

def create_prev5(match_id, team_id, teams):
    current_team = (teams[str(team_id)])
    team_string = current_team+"_stats.xlsx"
    t_df = pd.read_excel(team_string)
    col = list(t_df)
    #reversing allows us to find our current game and get the previous 5 in an easy way
    col.reverse()
    match_array = []
    j = 999
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
        if(i == match_id):
            ha_val = 0
            for element in t_df[i]:
                if(ha_val == 3):
                    y_label = element
                ha_val = ha_val + 1
            j = 0
    return match_array, y_label

#combine the things + current match metadata
#so it would go, array of metadata, append home_array, append away_array
def combine_prev5(home_id, away_id, round, home_array, away_array):
    current_example_array = [round, home_id, away_id]
    current_example_array.extend(home_array)
    current_example_array.extend(away_array)
    return current_example_array

def add_to_df(stat_df, example_m):
    #puts the current example of 10 prev games into ongoing df
    return stat_df

#assemebles a matrix which is nxm, and a related true labelled matrix of 1xm
# n = number of inputs, eg. stat categories of prev 5 games for each team + metadata for current game
# m = number of matches played from 2011 to current
#Loops through each match with the following 4 lines
#Should identify which teams are playing through FTP method
#Creates a home array and away aray of their previous 5 games through create_prev5
#combines these arrays into a 10 match array with the current match metadata through combine prev5
#adds this fully combined array into the ongoing dataframe of n*m, where m is amount of matches done
def assemble_stat_matrix(match_to_start_from, most_recent_match, teams, create_from_new):
    #Round 7 2011, as everyteam would have played 5 games by then.
    #create_from_new, is whether to re-make the whole data frame. 0 = re-make, 1=update
    i = match_to_start_from
    first = 0
    GWS = 1
    #for each match do determine teams, determine H/A create prev 5, combine the matches, add to big DF of example
    #while(i < 6330 or (i > 9297 and i < 9936 )):
    while(i<=most_recent_match):
        print(str(i))
        team1, team2 = find_teams_playing(i, teams)
        #takes into account GWS entering the league and not having 5 previous games
        if((team1 == 9 or team2 == 9) and GWS<6 and create_from_new == 0):
            GWS = GWS + 1
            i = i + 1
            continue
        #match doesn't exist
        if(team1 == 999 or team2 == 999):
            i = i + 1
            continue
        home_id, away_id, round = determine_home_away(i, team1, team2, teams)
        #made the create_prev5 function also return the h/a winloss value in another variable
        #It shouldn't matter that it finds it twice as it only adds it once
        #Should be 1xM, where M is the total matches found stats for.
        home_array, y_label = create_prev5(i, home_id, teams)
        away_array, y_label = create_prev5(i, away_id, teams)
        if(y_label == 0.5):
            y_label = 0
        current_example_array = combine_prev5(home_id, away_id, round, home_array, away_array)
        if(first == 0):
            #eg. we're only updating it
            if(create_from_new == 1):
                stats_df = pd.read_csv('assembled_stat_matrix.csv', index_col = 0)
                label_df = pd.read_csv('assembled_labelled_ymatrix.csv', index_col = 0)
                first = 1
                stats_df[str(i)] = current_example_array
                label_df[str(i)] = y_label
            else:
                data = {str(i) : current_example_array}
                label_data = {str(i): [y_label]}
                stats_df = pd.DataFrame(data)
                label_df = pd.DataFrame(label_data)
                first = 1
        else:
            stats_df[str(i)] = current_example_array
            label_df[str(i)] = y_label
        i = i + 1
    stats_df.to_csv('assembled_stat_matrix.csv')
    label_df.to_csv('assembled_labelled_ymatrix.csv')
    print(stats_df)
    print(label_df)

#updates current local files to the most recent versions
#have to manually check most recent game values though
#argv 1 is where to start from
#argv 2 is where to finish
#then assemebles stat matrices up until most recent values
def main():
    g = gad()
    teams = g.createTeamDict()
    #g.update(int(sys.argv[1]), int(sys.argv[2]),teams)
    assemble_stat_matrix(int(sys.argv[1]), int(sys.argv[2]), teams, 1)

if __name__ == '__main__':
    main()
