#uncomment below 2 lines, make this an argument to make this easier.
#Arg 1 = season
#Arg 2 = round

#Get these args from a py_script which goes to the season and round on footywire
  #and extracts match id to update from and to
  #python get_match_id_range.py
#Arg 3 = match ID to update from
#Arg 4 = match ID to update to

#get what the final lineups were like you usually run in the notebook
#python update_final_lineups.py $1 $2

Rscript R_Code/update.R $1 $2

python clean_team_data.py $3 $4

python assemble_df.py $3 $4 2
python assemble_df.py $3 $4 10
python assemble_df.py $3 $4 3


#python assemble_df.py 10615 4
#python assemble_df.py 10615 5
#python assemble_df.py 10615 6
#python assemble_df.py 10615 7
#python assemble_df.py 10615 8
#python assemble_df.py 10615 9
