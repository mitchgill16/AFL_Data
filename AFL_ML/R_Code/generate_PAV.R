### SOURCE: https://analysisofafl.netlify.app/paperthoughts/make-your-own-pav/
#! /usr/bin/Rscript

library('fitzRoy')
library('dplyr')
library('sys')

args = commandArgs(trailingOnly=TRUE)
getwd()
setwd("~/Documents/Mitch/AFL_Data/AFL_Data/AFL_ML/")
year <- as.integer(args[1])
#rnd <- as.integer(args[2])

#make Pavs for given year
#today <- Sys.Date()
afltables<-fetch_player_stats_afltables(year)

#footywire<-fitzRoy::fetch_player_stats_footywire(season = 2022)

afltables_home<-filter(afltables, Playing.for==Home.team)
afltables_away<-filter(afltables,Playing.for==Away.team)


afltables_home$pavO<-afltables_home$Home.score +
  0.25*afltables_home$Hit.Outs +
  3*afltables_home$Goal.Assists+
  afltables_home$Inside.50s+
  afltables_home$Marks.Inside.50+
  (afltables_home$Frees.For-afltables_home$Frees.Against)

afltables_home$pavD<-20*afltables_home$Rebounds +
  12*afltables_home$One.Percenters+
  (afltables_home$Marks-4*afltables_home$Marks.Inside.50+2*(afltables_home$Frees.For-afltables_home$Frees.Against))-
  2/3*afltables_home$Hit.Outs

afltables_home$pavM<-15*afltables_home$Inside.50s+
  20*afltables_home$Clearances +
  3*afltables_home$Tackles+
  1.5*afltables_home$Hit.Outs +
  (afltables_home$Frees.For-afltables_home$Frees.Against)



afltables_away$pavO<-afltables_away$Away.score +
  0.25*afltables_away$Hit.Outs +
  3*afltables_away$Goal.Assists+
  afltables_away$Inside.50s+
  afltables_away$Marks.Inside.50+
  (afltables_away$Frees.For-afltables_away$Frees.Against)


afltables_away$pavD<-20*afltables_away$Rebounds +
  12*afltables_away$One.Percenters+
  (afltables_away$Marks-4*afltables_away$Marks.Inside.50+2*(afltables_away$Frees.For-afltables_away$Frees.Against))-
  2/3*afltables_away$Hit.Outs



afltables_away$pavM<-15*afltables_away$Inside.50s+
  20*afltables_away$Clearances +
  3*afltables_away$Tackles+
  1.5*afltables_away$Hit.Outs +
  (afltables_away$Frees.For-afltables_away$Frees.Against)

fulltable<-rbind(afltables_home,afltables_away)
names(fulltable)

#check a player

players <- fulltable%>%group_by(Playing.for, First.name, Surname, ID)%>%
  summarise(total_O_pav=sum(pavO), total_D_pav=sum(pavD), total_M_pav=sum(pavM), games_played=n())

#players$total_pav <- players$total_O_pav + players$total_D_pav + players$total_M_pav

teams <- fulltable%>%group_by(Playing.for)%>%
  summarise(team_O_pav=sum(pavO), team_D_pav=sum(pavD), team_M_pav=sum(pavM))

sum_O <- sum(teams$team_O_pav)
sum_M <- sum(teams$team_M_pav)
sum_D <- sum(teams$team_D_pav)

teams$prop_pav_O <- (teams$team_O_pav / sum_O) * 1800
teams$prop_pav_M <- (teams$team_M_pav / sum_M) * 1800
teams$prop_pav_D <- (teams$team_D_pav / sum_D) * 1800

#teams$team_total_pav <- teams$team_O_pav + teams$team_M_pav + teams$team_D_pav

team_played <- fulltable%>%group_by(Playing.for, Round) %>% select(Playing.for, Round) %>% distinct() %>% 
  group_by(Playing.for) %>% summarise(team_played_games = n())

teams <- merge(teams, team_played)

combined_df <- left_join(players, teams, by = 'Playing.for')

combined_df$calculated_O_PAV <- (combined_df$total_O_pav / combined_df$team_O_pav) * combined_df$prop_pav_O

combined_df$calculated_M_PAV <- (combined_df$total_M_pav / combined_df$team_M_pav) * combined_df$prop_pav_M

combined_df$calculated_D_PAV <- (combined_df$total_D_pav / combined_df$team_D_pav) * combined_df$prop_pav_D

combined_df$calculated_total_PAV <- combined_df$calculated_D_PAV + combined_df$calculated_O_PAV + 
  combined_df$calculated_M_PAV

combined_df <- combined_df %>% select(Playing.for, First.name, Surname, calculated_total_PAV)
combined_df <- combined_df %>% rename(firstname = First.name, surname = Surname)

write.csv(combined_df, "R_Code/generated_pavs.csv", row.names=FALSE)
