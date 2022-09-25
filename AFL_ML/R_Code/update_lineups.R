#! /usr/bin/Rscript

library('fitzRoy')
library('dplyr')

args = commandArgs(trailingOnly=TRUE)
getwd()
year <- as.integer(args[1])
rnd <- as.integer(args[2])

x <- read.csv('R_Code/all_lineups.csv')
test <- fetch_lineup(source="AFL", season=year, round_number = rnd)
test <- select(test, teamName, round.roundNumber, player.playerName.givenName, player.playerName.surname)
test$year <- year
y <- test[c(5,1:4)]

#remove emergency check
for (team in c("Adelaide Crows", "Brisbane Lions", "Carlton", "Collingwood", "Essendon", "Fremantle",
               "Geelong Cats", "Gold Coast Suns", "GWS Giants", "Hawthorn", "Melbourne", "North Melbourne",
               "Port Adelaide", "Richmond", "St Kilda", "Sydney Swans", "West Coast Eagles", "Western Bulldogs")){
  print(team)
  z <- y[y$teamName == team,]
  if(dim(z)[1] > 22){
    z <- z[0:22,]
  }
  x <- rbind(x,z)
}


write.csv(x, "R_Code/all_lineups.csv", row.names=FALSE)
