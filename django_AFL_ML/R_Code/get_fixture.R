##get upcoming games
library('fitzRoy')
library('dplyr')

args = commandArgs(trailingOnly=TRUE)
getwd()
test_y <- as.integer(args[1])

fixture <- fetch_fixture(test_y) %>% select(round.roundNumber, home.team.club.name, away.team.club.name)

write.csv(fixture, "R_Code/fixture.csv", row.names=FALSE)