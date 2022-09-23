#to come back to when there is actually rounds to be upcoming

#Arg 1 = Current Season
#Arg 2 = Upcoming Round

#put a third arg here so we know not to bother updating ladder
Rscript R_Code/get_upcoming_games.R $1 $2
python predict.py $1 $2
