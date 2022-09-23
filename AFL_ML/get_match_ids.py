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
    bye_rounds = [i+1 for i, x in enumerate(counts) if x == 6]
    if(round_num < min(bye_rounds)):
        mids = df.iloc[(round_num*9)-9:(round_num*9)-1]
    elif(round_num >= min(bye_rounds) and round_num <= max(bye_rounds)):
        count = 0
        for x in bye_rounds:
            if(round_num == x):
                break
            count = count + 1
        mids = df.iloc[(round_num*9)-9-(3*count):((round_num)*9)-3-(3*count)]
    elif(round_num > max(bye_rounds) and round_num < 24):
        mids = df.iloc[((round_num-1)*9)-9:((round_num-1)*9)-1]
    elif(round_num == 24):
        mids = df.iloc[198:202]
    elif(round_num == 25):
        mids = df.iloc[202:204]
    elif(round_num == 26):
        mids = df.iloc[204:206]
    else:
        mids = df.iloc[206]
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
