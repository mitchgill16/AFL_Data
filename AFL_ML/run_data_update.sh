#uncomment below 2 lines, make this an argument to make this easier.
#Arg 1 = current season
#Arg 2 = previous round to gather data from

#Get these args from a py_script which goes to the season and round on footywire
  #and extracts match id to update from and to
python check_update.py $1 $2 || exit

python get_match_ids.py $1 $2

n=1
while IFS= read -r "variable$n"; do
  n=$((n + 1))
done < mids.txt

#get venue and ladder
Rscript R_Code/update.R $1 $2

#get what the final lineups were like you usually run in the notebook
python predict.py $1 $2

#web scrape the games data and clean the team sheets
python clean_team_data.py $variable1 $variable2

#append new data to dataset
python assemble_df.py $variable1 $variable2 2
python assemble_df.py $variable1 $variable2 10
python assemble_df.py $variable1 $variable2 3

#python assemble_df.py $4 4
#python assemble_df.py $4 5
#python assemble_df.py $4 6
#python assemble_df.py $4 7
#python assemble_df.py $4 8
#python assemble_df.py $4 9
