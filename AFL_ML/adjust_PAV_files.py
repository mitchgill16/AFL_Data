import pandas as pd
import numpy as np
import sys

def create_pav_new_dict():
    teamDict = {
        "1" : "Adelaide",
        "2" : "Brisbane Lions",
        "3" : "Carlton",
        "4" : "Collingwood",
        "5" : "Essendon",
        "6" : "Fremantle",
        "7" : "Geelong",
        "8" : "Gold Coast",
        "9" : "Greater Western Sydney",
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

def main():
    year = int(sys.argv[1])
    all_pavs = pd.read_csv("R_Code/all_player_PAVs.csv")
    new_pavs = pd.read_csv("R_Code/generated_pavs.csv")
    hpn = create_pav_new_dict()
    #reverse
    hpn = {y: x for x, y in hpn.items()}
    to_drop = all_pavs.index[(all_pavs['year'] == year)]
    all_pavs = all_pavs.drop(to_drop)
    #new_pavs = new_pavs.drop(['GM', 'Off PAV', 'Def PAV', 'Mid PAV',
    #                         'Off mPAV', 'Def mPAV', 'Mid mPAV', 'Total mPAV'], axis = 1)
    new_pavs['year'] = year
    #new_pavs[['surname', 'firstname']] = new_pavs['Player'].str.split(',', 1, expand=True)
    new_pavs['team'] = new_pavs['Playing.for'].map(hpn)
    new_pavs['PAV_total'] = new_pavs['calculated_total_PAV']
    new_pavs = new_pavs.drop(['Playing.for', 'calculated_total_PAV'], axis = 1)
    cols = ['team', 'year', 'firstname', 'surname', 'PAV_total']
    new_pavs = new_pavs[cols]
    all_pavs = pd.concat([all_pavs, new_pavs])
    all_pavs.to_csv("R_Code/all_player_PAVs.csv", header=True, index=False)

if __name__ == '__main__':
    main()
