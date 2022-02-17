### fetches venue for every game from 2013
library('fitzRoy')
library('dplyr')
year <- 2011
x <- 0
y <- 0
first <- 0
for(year in 2013:2021){
  print(year)
  if(first == 0){
    results <- fetch_results(year)
    first <- 1
    x <- select(results, round.year, round.roundNumber,
      match.homeTeam.name, match.awayTeam.name, venue.name)
    print(x)
  }
  else{
    results <- fetch_results(year)
    y <- select(results, round.year, round.roundNumber,
      match.homeTeam.name, match.awayTeam.name, venue.name)
    print(y)
    x <- rbind(x,y)
  }
}
write.csv(x, "2013_to_2021_venues.csv", row.names=FALSE)
