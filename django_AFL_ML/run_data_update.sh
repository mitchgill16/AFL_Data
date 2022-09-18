#uncomment below 2 lines, make this an argument to make this easier.
#Arg 1 = current season
#Arg 2 = previous round to gather data from

#Get these args from a py_script which goes to the season and round on footywire
  #and extracts match id to update from and to
  #python get_match_id_range.py
#Arg 3 = match ID to update from
#Arg 4 = match ID to update to

#get venue and ladder
Rscript R_Code/update.R $1 $2

#get what the final lineups were like you usually run in the notebook
python predict.py $1 $2

#web scrape the games data and clean the team sheets
python clean_team_data.py $3 $4

#append new data to dataset
python assemble_df.py $3 $4 2
python assemble_df.py $3 $4 10
python assemble_df.py $3 $4 3


#python assemble_df.py $4 4
#python assemble_df.py $4 5
#python assemble_df.py $4 6
#python assemble_df.py $4 7
#python assemble_df.py $4 8
#python assemble_df.py $4 9
