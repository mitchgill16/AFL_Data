import pandas as pd
import numpy as np

from Gather_AFL_Data import gatherer as gad

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
    print(df.shape)
    df.to_csv("Data/"+current_team+"_clean_stats.csv",index=True, header = True, index_label='Match_ID')


def main():
    g = gad()
    teams = g.createTeamDict()
    i = 1
    #should go through each of the 18 teams and create an excel file with over 100 stats for each game they played in
    while(i<19):
        clean_match_stats(teams,i)
        i = i+1

if __name__ == '__main__':
    main()
