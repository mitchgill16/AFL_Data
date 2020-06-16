#to do: make the dir import work, to have more methods available

import numpy as np
import pandas as pd
import random as rand
from Gather_AFL_Data import gatherer as gad

def find_teams_playing(match_id, teams):
    i = 1
    match_id = str(match_id)
    team1 = 999
    team2 = 999
    while(i<19):
        current_team = (teams[str(i)])
        print(current_team)
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
            j = 0
    return match_array

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

#assemebles a matrix which is nxm
# n = number of inputs, eg. stat categories of prev 5 games for each team + metadata for current game
# m = number of matches played from 2011 to current
#Loops through each match with the following 4 lines
#Should identify which teams are playing through FTP method
#Creates a home array and away aray of their previous 5 games through create_prev5
#combines these arrays into a 10 match array with the current match metadata through combine prev5
#adds this fully combined array into the ongoing dataframe of n*m, where m is amount of matches done
def assemble_stat_matrix():
    #gets team dictionary
    teams = gad.createTeamDict()
    #for each match do determine teams, determine H/A create prev 5, combine the matches, add to big DF of example
    team1, team2 = find_teams_playing(9913, teams)
    home_id, away_id, round = determine_home_away(9913, team1, team2, teams)
    home_array = create_prev5(9913, home_id, teams)
    away_array = create_prev5(9913, away_id, teams)
    current_example_array = combine_prev5(home_id, away_id, round, home_array, away_array)
    print(current_example_array)

def main():
    #could make parameters a range of numbers?
    assemble_stat_matrix()

if __name__ == '__main__':
    main()
