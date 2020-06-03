# To do: get the match ID for each team in a text file.
# get match stats for each game for each team

import sys
import requests
import pprint
from bs4 import BeautifulSoup

#Scrapes webpage for which teams played
#inputs are a team dictionary the team we are looking at and the match num
def scrape_match_teams_playing(teams, team_id, match_num):
    flag = 0
    team = teams.get(str(team_id))
    URL = "https://www.footywire.com/afl/footy/ft_match_statistics?mid=" + str(match_num)
    #print(URL)
    current_team = (teams[str(team_id)])
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    #returns the teams playing for each match by getting the text element
    #of the HTML returned by the soup object
    data = [element.text for element in soup.find_all('td', class_='bnorm', width='190')]
    #print(data)
    #match_data = results.find_all('td', class_="statdata")
    if current_team in data:
        flag = 1
    return flag

#Scrapes webpage for advanced stats match data for a given team and match
#def scrape_match_advanced_stats(teams, team_id, match_num):
#    team = teams.get(str(team_id))
#    URL = "https://www.footywire.com/afl/footy/ft_match_statistics?mid="+str(match_num)+"&advv=Y"
#    print(URL)
#    page = requests.get(URL)
    #print(page.content)
#    soup = BeautifulSoup(page.content, 'html.parser')

#creates a dictionary with each teams identifier on afl_tables
def createTeamDict():
    teamDict = {
    "1" : "Adelaide",
    "2" : "Brisbane",
    "3" : "Carlton",
    "4" : "Collingwood",
    "5" : "Essendon",
    "6" : "Fremantle",
    "7" : "Geelong",
    "8" : "Gold Coast",
    "9" : "GWS",
    "10": "Hawthorn",
    "11": "Melbourne",
    "12": "North Melbourne",
    "13": "Port Adelaide",
    "14": "Richmond",
    "15": "St Kilda",
    "16": "Sydney",
    "17": "West Coast",
    "18": "Western Bulldogs"
    }
    return teamDict

def createTeamMatchFile(team_int, team_dict):
    #gets current team we are looking at
    current_team = (team_dict[str(team_int)])
    print(current_team)
    textfile = open(current_team+"_data.txt", 'w+')
    match = 1
    #round 1 2011, start of the expansion teams
    p = 5147
    while(p<6362):
        #checks if current game has current team
        flag_num = scrape_match_teams_playing(team_dict, team_int, p)
        #if it does it records the match ID, makes it easier for the future
        if(flag_num == 1):
            #matches played because it doesn't take into account the bye round
            print("Match: "+str(match))
            match = match + 1
            textfile.write(str(p)+'\n')
        p = p+1
    #match starts at EF 2016 WC vs WB
    j = 9298
    #End of Round 1 2020
    while(j<9936):
        #checks if current game has current team
        flag_num = scrape_match_teams_playing(team_dict, team_int, j)
        #if it does it records the match ID, makes it easier for the future
        if(flag_num == 1):
            #matches played because it doesn't take into account the bye round
            print("match: "+str(match))
            match = match + 1
            textfile.write(str(j)+'\n')
        j = j+1
    textfile.close()

def main():
    teams = createTeamDict()
    i = 7
    #should go through each of the 18 teams and create a txt file of their match ID's
    while(i<19):
        createTeamMatchFile(i,teams)
        i = i+1

if __name__ == '__main__':
    main()
