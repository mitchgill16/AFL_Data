import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import subprocess
import sys

def main():
    season = int(sys.argv[1])
    round_num = int(sys.argv[2])
    URL = "https://www.footywire.com/afl/footy/ft_match_list?year="+str(season)
    #print(URL)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = [element['href'] for element in soup.select("a[href*=mid]")]
    df = pd.DataFrame({"urls" : data})
    df['urls'] = df['urls'].str[24:]
    subprocess.call(["/usr/bin/Rscript", "R_Code/get_fixture.R", str(season)])
    fixture = pd.read_csv("R_Code/fixture.csv")
    counts = fixture['round.roundNumber'].value_counts()
    counts = counts.sort_index()
    bye_rounds = [i+1 for i, x in enumerate(counts) if (x >= 4 and x < 9)]
    if(round_num < min(bye_rounds) and season < 2024):
        mids = df.iloc[(round_num*9)-9:(round_num*9)]
    elif(round_num == 0 and season == 2024):
        mids = df.iloc[0:4]
    elif(round_num == 1 and season == 2024):
        mids = df.iloc[4:13]
    elif(round_num == 2 and season == 2024):
        mids = df.iloc[13:21]
    elif(round_num == 3 and season == 2024):
        mids = df.iloc[21:29]
    elif(round_num == 4 and season == 2024):
        mids = df.iloc[29:38]
    elif(round_num == 5 and season == 2024):
        mids = df.iloc[38:46]
    elif(round_num == 6 and season == 2024):
        mids = df.iloc[46:54]
    elif(round_num >= 7 and round_num < 12 and season == 2024):
        mids = df.iloc[(round_num-1)*9:round_num*9]
    elif(round_num == 12 and season == 2024):
        mids = df.iloc[99:106]
    elif(round_num == 13 and season == 2024):
        mids = df.iloc[106:114]
    elif(round_num == 14 and season == 2024):
        mids = df.iloc[114:120]
    elif(round_num == 15 and season == 2024):
        mids = df.iloc[120:126]

    elif(round_num >= min(bye_rounds) and round_num <= max(bye_rounds) and season < 2023):
        count = 0
        for x in bye_rounds:
            if(round_num == x):
                break
            count = count + 1
        mids = df.iloc[(round_num*9)-9-(3*count):((round_num)*9)-3-(3*count)]
    elif(round_num >= min(bye_rounds) and round_num <= max(bye_rounds) and season == 2023):
        #cbf doing logic, just manual enter for bye rounds
        if(round_num == 12):
            mids = df.iloc[99:106]
        elif(round_num == 13):
            mids = df.iloc[106:114]
        elif(round_num == 14):
            mids = df.iloc[114:120]
        elif(round_num == 15):
            mids = df.iloc[120:126]
    elif(round_num > max(bye_rounds) and round_num < 24 and season < 2023):
        mids = df.iloc[((round_num-1)*9)-9:((round_num-1)*9)]
    elif(round_num > max(bye_rounds) and round_num < 25 and season >= 2023):
        mids = df.iloc[((round_num-1)*9)-9:((round_num-1)*9)]
    elif(season < 2023):
        if(round_num == 24):
            mids = df.iloc[198:202]
        elif(round_num == 25):
            mids = df.iloc[202:204]
        elif(round_num == 26):
            mids = df.iloc[204:206]
        else:
            mids = df.iloc[206:207]
    #2023 onwards
    elif(season > 2022):
        if(round_num == 25):
            mids = df.iloc[207:211]
        elif(round_num == 26):
            mids = df.iloc[211:213]
        elif(round_num == 27):
            mids = df.iloc[213:215]
        else:
            mids = df.iloc[215:216]
    min_mid = (min(mids['urls']))
    max_mid = (max(mids['urls']))
    print(min_mid)
    print(max_mid)
    with open("mids.txt", "w") as text_file:
        text_file.write(min_mid)
        text_file.write("\n")
        text_file.write(max_mid)
        text_file.close()

if __name__ == '__main__':
    main()
