##update function
library('fitzRoy')
library('dplyr')

args = commandArgs(trailingOnly=TRUE)
getwd()
test_y <- as.integer(args[1])
test_r <- as.integer(args[2])

all_ladders <- read.csv("R_Code/all_ladders.csv")
all_venues <- read.csv("R_Code/all_venues.csv")

results <- fetch_results(test_y, test_r)
venue <- select(results, round.year, round.roundNumber,
            match.homeTeam.name, match.awayTeam.name, venue.name)
combined_venues <- rbind(all_venues, venue) %>% distinct()

combined_venues$match.homeTeam.name <- gsub("SUNS", "Suns", combined_venues$match.homeTeam.name)
combined_venues$match.homeTeam.name <- gsub("GIANTS", "Giants", combined_venues$match.homeTeam.name)
combined_venues$match.awayTeam.name <- gsub("SUNS", "Suns", combined_venues$match.awayTeam.name)
combined_venues$match.awayTeam.name <- gsub("GIANTS", "Giants", combined_venues$match.awayTeam.name)

write.csv(combined_venues, "R_Code/all_venues.csv", row.names=FALSE)

if(test_r < 24 | (test_r < 25 & test_y > 2022)){
  ladders <- fetch_ladder(season=test_y, round=test_r)
  ladder <- select(ladders, season, round_number, team.name, position,form,
              thisSeasonRecord.winLossRecord.wins, thisSeasonRecord.winLossRecord.losses,
              thisSeasonRecord.winLossRecord.draws)
  combined_ladders <- rbind(all_ladders, ladder) %>% distinct()
  write.csv(combined_ladders, "R_Code/all_ladders.csv", row.names=FALSE)
}
