import pandas as pd
import numpy as np
import sys

from Gather_AFL_Data import gatherer as gad

#makes a combined dataframe of each team and their team_match_ids
#saves doing a loop through each file like old design
def clean_team_ids():
    g = gad()
    teams = g.createTeamDict()

    all_data = []
    team_list = []

    i = 1
    while(i<19):
        current_team = (teams[str(i)])
        team_list.append(current_team)
        textfile = open("Data/"+current_team+"_data.txt", 'r')
        lines = textfile.readlines()
        lines = [x.strip() for x in lines]
        all_data.append(lines)
        i = i + 1
    print(len(all_data))
    team_list
    df = pd.DataFrame.from_records(all_data)
    df = df.T
    df.to_csv("Data/teams_match_ids.csv", header=True, index=False)

#goes through match files with scraped data from footywire
#transposes it to how it should be eg. rows = samples, columns = inputs
#removes duplicate data
#sorts based on year and round rather than match_id
def clean_match_stats(team_dict, team_int):
    current_team = (team_dict[str(team_int)])
    print(current_team)
    df = pd.read_excel("Data/"+current_team+'_stats.xlsx')
    df= df.drop(df.filter(regex='\.\d').columns, axis=1)
    print(df.dtypes)
    #df = df.drop_duplicates()
    df = df.T
    print(df.tail())
    df.columns = df.iloc[0]
    df = df[1:]
    headers = []
    to_remove = []
    for x in range(0, len(df.columns)):
        #print(x)
        y = df.columns[x]
       # print(y[-1])
        headers.append(df.columns[x])
        if(y[-1] == '2'):
            #print(y)
            to_remove.append(y)
        if(y[-2] == 'O'):
            if(y[-4] == '2'):
                #print(y)
                to_remove.append(y)
    df = df.drop(to_remove,axis=1)
    print(df.shape)
    df = df.drop_duplicates()
    print(df.shape)
    #remove rows with years from 2023 onwards as there is 24 rounds
    year_idx = df.loc[(df['Year'] > 2022)].index
    year_df = df.loc[(df['Year'] > 2022)]
    df.drop(year_idx, inplace=True)
    #fix my finals + 1 error which I added for some reason
    idx = df.loc[(df['Round'] > 24)].index
    finals_df = df.loc[(df['Round'] > 24)]
    df.drop(idx,inplace=True)
    finals_df['Round'] = finals_df['Round'] - 1
    df = pd.concat([df, finals_df], ignore_index = False)
    df = pd.concat([df, year_df], ignore_index = False)
    s_df = df.sort_values(["Year", "Round"], ascending = (True, True))
    print("Dataframe was already sorted: "+str(s_df.equals(df)))
    s_df.to_csv("Data/"+current_team+"_clean_stats.csv",index=True, header = True, index_label='Match_ID')

#footywire team names and code gathered through R is different identifiers
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

def create_venue_alias_dict():
    vDict = {
        "Heritage Bank Stadium" : "Metricon Stadium",
        "UTAS Stadium" : "University of Tasmania Statium",
        "AAMI Stadium" : "Adelaide Oval",
        "Domain Stadium" : "Optus Stadium"
        }
    return vDict

##TO Finish here -- ladder pos
##crops the clean data to be from 2013 onwards
##adds venue and ladder to cropped team data by appending column
##Accepts team_dict from footywire, R code and current team int
def append_r_data(team_dict, r_dict, team_int):
    current_team = (team_dict[str(team_int)])
    print(current_team)
    #load data
    df = pd.read_csv("Data/"+current_team+'_clean_stats.csv')
    venue = pd.read_csv("R_Code/all_venues.csv")
    ladders = pd.read_csv("R_Code/all_ladders.csv")
    current_r_team = (r_dict[str(team_int)])
    team_venue = venue[venue.isin([current_r_team]).any(axis=1)]

    #alias change venue names
    vdict = create_venue_alias_dict()

    #ladder data preprocess
    team_ladders = ladders[ladders.isin([current_r_team]).any(axis=1)]
    team_ladders.season = team_ladders.season.astype(float)
    team_ladders.round_number = team_ladders.round_number.astype(float)
    team_ladders.rename(columns={"season": "Year", "round_number": "Round"}, inplace=True)

    #pav data preprocess
    pavs = pd.read_csv("R_Code/all_team_pavs.csv")
    team_pavs = pavs.loc[(pavs["Team_ID"]==team_int)]

    #add venue and slice original dataframe
    tv_rows = team_venue.shape[0]
    #here would be a good point to exclude any game in 2020?
    sliced_df = df[-tv_rows:]
    #add PAV total and venue name for the matches
    sliced_df['PAV_Sum'] = team_pavs['Player_PAV_Total'].to_numpy()
    venues = team_venue['venue.name'].to_numpy()
    v_int = 0
    while v_int < len(venues):
        current_ven = venues[v_int]
        if(current_ven in vdict):
            ven_alias = vdict.get(current_ven)
            print("current venue: " + str(current_ven) + " changed to: " +str(ven_alias))
            venues[v_int] = ven_alias
        v_int = v_int + 1
    sliced_df['Venue'] = venues
    print(sliced_df.iloc[0][1:3])

    #merge ladder information into dataset, remove duplicate team name
    #fills forward missing finals data with most recent season data
    merged_df = sliced_df.merge(team_ladders, how='left', on=['Year', 'Round'])
    merged_df.drop(['team.name'], inplace=True, axis = 1)
    merged_df.fillna(method='ffill', inplace=True)

    #saves data
    merged_df.to_csv("Data/"+current_team+"_clean_stats.csv",header = True, index=False)

def main():
    g = gad()
    teams = g.createTeamDict()
    g.update(int(sys.argv[1]), int(sys.argv[2]),teams)
    r_teams = create_R_TeamDict()
    i = 1
    #should go through each of the 18 teams and create an excel file with over 100 stats for each game they played in
    while(i<19):
        clean_match_stats(teams,i)
        append_r_data(teams, r_teams, i)
        i = i+1
    clean_team_ids()

if __name__ == '__main__':
    main()
