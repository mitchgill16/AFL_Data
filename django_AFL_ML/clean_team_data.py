import pandas as pd
import numpy as np

from Gather_AFL_Data import gatherer as gad

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

def clean_match_stats(team_dict, team_int):
    current_team = (team_dict[str(team_int)])
    print(current_team)
    df = pd.read_excel("Data/"+current_team+'_stats.xlsx')
    df = df.T
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
    s_df = df.sort_values(["Year", "Round"], ascending = (True, True))
    print("Dataframe was already sorted: "+str(s_df.equals(df)))
    s_df.to_csv("Data/"+current_team+"_clean_stats.csv",index=True, header = True, index_label='Match_ID')


def main():
    g = gad()
    teams = g.createTeamDict()
    i = 1
    #should go through each of the 18 teams and create an excel file with over 100 stats for each game they played in
    while(i<19):
        clean_match_stats(teams,i)
        i = i+1
    clean_team_ids()

if __name__ == '__main__':
    main()
