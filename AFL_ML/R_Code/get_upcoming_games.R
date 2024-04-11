##get upcoming games
library('fitzRoy')
library('dplyr')

args = commandArgs(trailingOnly=TRUE)
getwd()
test_y <- as.integer(args[1])
test_r <- as.integer(args[2])

all_venues <- read.csv("R_Code/all_venues.csv")

results <- fetch_fixture(test_y, test_r)
venue <- select(results, utcStartTime, round.roundNumber,
                home.team.name, away.team.name, venue.name)
venue$utcStartTime <- substr(venue$utcStartTime,0,4)
venue <- venue %>% rename(round.year = utcStartTime, match.homeTeam.name = home.team.name, match.awayTeam.name = away.team.name)

venue$match.homeTeam.name <- gsub("SUNS", "Suns", venue$match.homeTeam.name)
venue$match.homeTeam.name <- gsub("GIANTS", "Giants", venue$match.homeTeam.name)
venue$match.awayTeam.name <- gsub("SUNS", "Suns", venue$match.awayTeam.name)
venue$match.awayTeam.name <- gsub("GIANTS", "Giants", venue$match.awayTeam.name)


print(venue)

combined_venues <- rbind(all_venues, venue) %>% distinct()

combined_venues$match.homeTeam.name <- gsub("SUNS", "Suns", combined_venues$match.homeTeam.name)
combined_venues$match.homeTeam.name <- gsub("GIANTS", "Giants", combined_venues$match.homeTeam.name)
combined_venues$match.awayTeam.name <- gsub("SUNS", "Suns", combined_venues$match.awayTeam.name)
combined_venues$match.awayTeam.name <- gsub("GIANTS", "Giants", combined_venues$match.awayTeam.name)

write.csv(combined_venues, "R_Code/all_venues.csv", row.names=FALSE)
