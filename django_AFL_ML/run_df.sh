#uncomment below 2 lines, make this an argument to make this easier.

Rscript R_Code/update.R 2022 1
python Gather_AFL_Data.py 10544 10552

python clean_team_data.py

python assemble_df.py 10552 1
python assemble_df.py 10552 10

#python assemble_df.py 10544 10552 3
#python assemble_df.py 10544 10552 4
#python assemble_df.py 10544 10552 5
#python assemble_df.py 10544 10552 6
#python assemble_df.py 10544 10552 7
#python assemble_df.py 10544 10552 8
#python assemble_df.py 10544 10552 9
#python assemble_df.py 10544 10552 2
