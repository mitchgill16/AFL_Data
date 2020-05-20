import sys
import requests
from bs4 import BeautifulSoup
#Scrapes webpage for data
#inputs are a team dictionary and a preamble of URL string
def scrape_webpage(teams,team_id):
    team = teams.get(str(team_id))
    #URL = "http://afltables.com/afl/teams/"+team+"/overall_wl.html"
    URL = "https://www.footywire.com/afl/footy/th-"+team
    print(URL)
    page = requests.get(URL)
#    soup = BeautifulSoup(page.content, 'html.parser')
#    past5 = soup.find(id = 'sortableTable0')
    print(page.status_code)

    #test teamdict works

#creates a dictionary with each teams identifier on afl_tables
def createTeamDict():
    teamDict = {
    "1" : "adelaide-crows",
    "2" : "brisbane-lions",
    "3" : "carlton-blues",
    "4" : "collingwood-magpies",
    "5" : "essendon-bombers",
    "6" : "fremantle-dockers",
    "7" : "geelong-cats",
    "8" : "gold-coast-suns",
    "9" : "greater-western-sydney-giants",
    "10": "hawthorn-hawks",
    "11": "melbourne-demons",
    "12": "kangaroos",
    "13": "port-adelaide-power",
    "14": "richmond-tigers",
    "15": "st-kilda-saints",
    "16": "sydney-swans",
    "17": "west-coast-eagles",
    "18": "western-bulldogs"
    }
    return teamDict

def main():
    teams = createTeamDict()
    i = 1
    while(i<19):
        scrape_webpage(teams,i)
        i = i+1

if __name__ == '__main__':
    main()
