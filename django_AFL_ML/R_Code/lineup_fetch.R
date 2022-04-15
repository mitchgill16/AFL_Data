library('fitzRoy')
library('dplyr')

x <- 0
y <- 0
first <- 0

for(year in 2022:2022){
  rnd <- 1
  print(year)
  for (rnd in 1:4){
    print(rnd)
    if(first == 0){
      test <- fetch_lineup(source="AFL", season=year, round_number = rnd)
      test <- select(test, teamName, round.roundNumber, player.playerName.givenName, player.playerName.surname)
      test$year <- year
      x <- test[c(5,1:4)]
      first <- 1
    }
    else{
      if(year == 2020){
        if(rnd > 22){
          break
        }
      }
      test <- fetch_lineup(source="AFL", season=year, round_number = rnd)
      test <- select(test, teamName, round.roundNumber, player.playerName.givenName, player.playerName.surname)
      test$year <- year
      y <- test[c(5,1:4)]
      x <- rbind(x,y)
    }
  }
}

write.csv(x, "/home/chris/Documents/Mitch/AFL_Data/AFL_Data/django_AFL_ML/R_Code/all_lineups.csv", row.names=FALSE)
