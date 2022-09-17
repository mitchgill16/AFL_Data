#Takes the information created from Gather_AFL_Data.py and
#cleans the data with clean_team_data.py
#produces the input matrix (X) and the labelled array (Y)
#in a format that should be suitable for deep learning packages
#Matrix X is of NxM where N is number of rows.
#N is each sample match to be predicted with previous n_games of stats
#M is each input
#Array Y should be of 1xM, where each value is whether the home or away team Won
#Margin array also for the margin from home team perspective (negative = loss)

import numpy as np
import pandas as pd
import random as rand
import sys
from Gather_AFL_Data import gatherer as gad
import clean_team_data

#takes a match_id and two team ids
#looks into the team_match_ids spreadsheet
#returns col_idx which is the team id's
def find_teams_playing(match_id, teams):
    match_id = str(match_id)
    df = pd.read_csv("Data/teams_match_ids.csv")
    match_teams = ([col for col in df.columns if df[col].apply(str).str.contains(match_id).any()])
    match_teams = [int(i) for i in match_teams]
    match_teams = [x+1 for x in match_teams]
    print(match_teams)
    if(len(match_teams) != 2):
        print("REEEEEEEEEEEEEE")
        team1 = 999
        team2 = 999
    else:
        team1 = match_teams[0]
        team2 = match_teams[1]
    return team1, team2

#looks into the clean_stats.csv for 2 given teams
#looks in the H/A category to figure out home/away teams
#also returns the current round for the match_id
def determine_home_away(match_id, team1, team2, teams):
    #gets current team string and their stats file
    current_team = (teams[str(team1)])
    team_string = current_team+"_clean_stats.csv"
    #reads the team1 excel into a dataframe, sets match_id as an index
    t1_df = pd.read_csv("Data/"+team_string, index_col="Match_ID")
    #turns index into str
    t1_df.index = t1_df.index.map(int)
    t1_df.index = t1_df.index.map(str)
    match_id = str(match_id)
    #print(match_id)
    #retrieves the ha_val and round from first team excel
    #t1_df['Match_ID'] = t1_df['Match_ID'].astype(int)
    ha_val = t1_df.loc[match_id,"H/A?"]
    round_val = t1_df.loc[match_id, "Round"]
    venue = t1_df.loc[match_id, "Venue"]
    pav = t1_df.loc[match_id, "PAV_Sum"]
    #print(ha_val)
    #logic that if first team is 0, then other team must be 1
    if(ha_val == 0.0):
        #print('ha_val is 0')
        home_id = team1
        away_id = team2
    elif(ha_val == 1.0):
        #print('ha_val is 1')
        home_id = team2
        away_id = team1
    #quick indicator if somethings happened
    else:
        #somethings gone wrong
        home_id = 999
        away_id = 999
    return home_id, away_id, round_val, venue, pav

#finds n_games previous worth of data for a given team
#takes match_id to look back n_games from
def create_prev_games(match_id, team_id, teams, flag, n_games):
    margin = None
    ma = None
    y_label = None
    current_team = (teams[str(team_id)])
    print(current_team)
    print(match_id)
    team_string = current_team+"_clean_stats.csv"
    df = pd.read_csv("Data/"+team_string)
    #drops ladder stats
    #finds where in the dataframe the current match is
    idx = df.index[df['Match_ID'] == match_id]
    #print(idx)
    my_idx = idx[0]
    #splits dataframe into game data and end of round ladder data
    l_df = df.iloc[:,-5:]
    t_df = df.iloc[: , :-5]
    current_year = t_df.loc[my_idx][1]
    if(current_year == 2020.0):
        print('game in 2020')
        margin = 8888
    else:
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
        if(my_idx < (n_games)):
            print('Num of Prev Games Exceeds previous games')
            margin = 9999
        else:
            #start match array with the ladder values from end of previous round (as this would be current for predicting round)
            ma = l_df.loc[my_idx-1].values
            #finds both labels for models
            y_label = t_df.loc[my_idx]["H/A Win?"]
            margin = t_df.loc[my_idx]["Margin"]
            #start from the previous game to current game
            #i is to know how many games included
            i = 1
            #j finds the previous game and allows for 2020 exclusion
            j = 1
            while i <= n_games:
                year = t_df.loc[my_idx-j][1]
                if(year == 2020.0):
                    j = j + 1
                    continue
                cg = t_df.loc[my_idx-j][2:].values
                ma = [*ma, *cg]
                i = i + 1
                j = j + 1
    return ma, y_label, margin

#combine the things + current match metadata
#so it would go, array of metadata, append home_array, append away_array
def combine_prev_games(home_id, away_id, round, venue, h_pav, a_pav, home_array, away_array):
    current_example_array = [round, home_id, away_id, venue, h_pav, a_pav]
    current_example_array.extend(home_array)
    current_example_array.extend(away_array)
    return current_example_array

#return an array with headers
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

#assemebles a matrix that goes through an either creates a new Matrix to a given match_id
#or appends data between inclusive two match ID's
def assemble_stat_matrix(match_to_start_from, most_recent_match, teams, n_games, new):
    #round 1 2012 (5343)
    #Round 6 2012 (5388),
    #Round 1 2013 (5550)
    #to current 10543 as everyteam would have played 5 games by then.
    #create_from_new, is whether to re-make the whole data frame. 0 = re-make, 1=update
    i = match_to_start_from
    stats_df = []
    label_df = []
    margin_df = []
    while(i<=most_recent_match):
        print(str(i))
        team1, team2 = find_teams_playing(i, teams)
        if(team1 == 999 or team2 == 999):
            print('no_match exist')
            i = i + 1
            continue
        home_id, away_id, round, venue, h_pav = determine_home_away(i, team1, team2, teams)
        #too lazy to update the function, easier to just to flip to get a_pav
        aaaa, bbbb, cccc, dddd, a_pav = determine_home_away(i, team2, team1, teams)
        #check to see if the a_pav and h_pav need to be flipped
        if(team1 == away_id):
            temp = h_pav
            h_pav = a_pav
            a_pav = temp
        #made the create_prev5 function also return the h/a winloss value in another variable
        #It shouldn't matter that it finds it twice as it only adds it once
        #margin will be perspective of home team, however will be abs for regression
        #Should be 1xM, where M is the total matches found stats for.
        away_array, y_label, margin = create_prev_games(i, away_id, teams, 1, n_games)
        #checks to make sure you an actually get enough previous game data
        if(margin == 8888):
            print('current game is in 2020')
            i = i + 1
            continue
        if(margin == 9999):
            print('too little previous examples for n_games')
            i = i + 1
            continue
        home_array, y_label, margin = create_prev_games(i, home_id, teams, 0, n_games)
        if(margin == 9999):
            print('too little previous examples for n_games')
            i = i + 1
            continue
        if(y_label == 0.5):
            y_label = 0
        current_example_array = combine_prev_games(home_id, away_id, round, venue, h_pav, a_pav, home_array, away_array)
        stats_df.append(current_example_array)
        margin_df.append(margin)
        label_df.append(y_label)
        i = i + 1
    #create new df and save
    if(new):
        stats_df = pd.DataFrame(stats_df)
        h = get_headers(n_games)
        stats_df.columns = h
        stats_df.to_csv('Data/assembled_stat_matrix_no2020'+str(n_games)+'_games.csv', index = False)

        label_df = pd.DataFrame(label_df)
        label_header = ['H/A Win?']
        label_df.columns = label_header
        label_df.to_csv('Data/assembled_labelled_ymatrix_no2020'+str(n_games)+'_games.csv', index =False)

        margin_df = pd.DataFrame(margin_df)
        margin_header = ['Margin']
        margin_df.columns = margin_header
        margin_df.to_csv('Data/assembled_margin_ymatrix_no2020'+str(n_games)+'_games.csv', index = False)
    #append created df to prevously saved df
    else:
        s_df = pd.read_csv('Data/assembled_stat_matrix_no2020'+str(n_games)+'_games.csv')
        h = get_headers(n_games)
        stats_df = pd.DataFrame(stats_df)
        stats_df.columns = h
        s_df = pd.concat([s_df, stats_df], ignore_index = True)
        idx = s_df[s_df.duplicated()]
        idx = idx.index
        s_df.drop(idx,inplace=True)
        s_df.to_csv('Data/assembled_stat_matrix_no2020'+str(n_games)+'_games.csv', index = False)

        l_df = pd.read_csv('Data/assembled_labelled_ymatrix_no2020'+str(n_games)+'_games.csv')
        label_df = pd.DataFrame(label_df)
        l_h = ['H/A Win?']
        label_df.columns = l_h
        l_df = pd.concat([l_df, label_df], ignore_index = True)
        l_df.drop(idx,inplace=True)
        l_df.to_csv('Data/assembled_labelled_ymatrix_no2020'+str(n_games)+'_games.csv', index =False)

        m_df = pd.read_csv('Data/assembled_margin_ymatrix_no2020'+str(n_games)+'_games.csv')
        margin_df = pd.DataFrame(margin_df)
        m_h = ['Margin']
        margin_df.columns = m_h
        m_df = pd.concat([m_df, margin_df], ignore_index = True)
        m_df.drop(idx,inplace=True)
        m_df.to_csv('Data/assembled_margin_ymatrix_no2020'+str(n_games)+'_games.csv', index = False)

    print(stats_df)
    print(label_df)
    print(margin_df)

#updates current local files to the most recent versions
#have to manually check most recent game values though
#argv 1 is where to start from
#argv 2 is where to finish
#or
#argv 1 is where to finish if just starting again, as it will assume new df

def main():
    g = gad()
    teams = g.createTeamDict()
    #5388 = first game Round 7 2012
    #5550 = first game round 1 2013
    if(len(sys.argv) == 3):
        new = True
        assemble_stat_matrix(5550, int(sys.argv[1]), teams, int(sys.argv[2]), new)
    else:
        new = False
        assemble_stat_matrix(int(sys.argv[1]), int(sys.argv[2]), teams, int(sys.argv[3]), new)

if __name__ == '__main__':
    main()
