import pandas as pd
import numpy as np
import sys

def main():
    year = int(sys.argv[1])
    round_num = int(sys.argv[2])
    venue = pd.read_csv("R_Code/all_venues.csv")
    pavs = pd.read_csv("R_Code/all_team_pavs.csv")
    year_venue = venue.loc[(venue['round.year'] == year)]
    year_pav = pavs.loc[(pavs['Year'] == year)]
    if((max(venue['round.year']) > year) or (max(pavs['Year']) > year) ):
        raise ValueError("Update Error... \n Round already updated Update for this given round and year has already been completed and has been used to generate future predictions. Please delete future rounds and rows from R_Code/all_venues.csv and/or R_Code/all_team_pavs.csv if this was not a mistake")
        sys.exit(1)
    elif((max(venue['round.year']) == year) and (max(year_venue['round.roundNumber']) > round_num)):
        raise ValueError("Update Error... \n Round already updated Update for this given round and year has already been completed and has been used to generate future predictions. Please delete future rounds and rows from R_Code/all_venues.csv and/or R_Code/all_team_pavs.csv if this was not a mistake")
        sys.exit(1)
    elif((max(pavs['Year']) == year) and (max(year_pav['Round']) > round_num)):
        raise ValueError("Update Error... \n Round already updated Update for this given round and year has already been completed and has been used to generate future predictions. Please delete future rounds and rows from R_Code/all_venues.csv and/or R_Code/all_team_pavs.csv if this was not a mistake")
        sys.exit(1)

if __name__ == '__main__':
    main()
