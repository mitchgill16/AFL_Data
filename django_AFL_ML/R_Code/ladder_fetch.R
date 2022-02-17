### some sort of loop that goes through each teams clean_stats
### appends a ladder position column (R1 = 0)
### Can be called from clean data
library('fitzRoy')
library('dplyr')
year <- 2011
x <- 0
y <- 0
first <- 0
for(year in 2013:2021){
  rnd <- 1
  print(year)
  for (rnd in 1:23){
    print(rnd)
    if(first == 0){
      ladder <- fetch_ladder(season=year, round=rnd)
      x <- select(ladder, season, round_number, team.name, position,form,
        thisSeasonRecord.winLossRecord.wins, thisSeasonRecord.winLossRecord.losses,
          thisSeasonRecord.winLossRecord.draws)
      print(x)
      first <- 1
    }
    else{
      if(year == 2020){
        if(rnd > 18){
          break
        }
      }
      ladder <- fetch_ladder(season=year, round=rnd)
      y <-select(ladder, season, round_number, team.name, position,form,
        thisSeasonRecord.winLossRecord.wins, thisSeasonRecord.winLossRecord.losses,
          thisSeasonRecord.winLossRecord.draws)
      print(y)
      x <- rbind(x,y)
    }
  }
}
write.csv(x, "all_ladders.csv", row.names=FALSE)

#do same for venue
