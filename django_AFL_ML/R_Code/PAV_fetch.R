library('fitzRoy')
library('dplyr')

x <- 0
y <- 0
first <- 0

setwd("/home/chris/Documents/Mitch/AFL_Data/AFL_Data/django_AFL_ML/R_Code/")

for(year_num in 2012:2019){
  print(year_num)
  for(team_id in 1:18){
    print(team_id)
    if(first == 0){
      results <- fetch_squiggle_data("pav", year=year_num, team=team_id)
      first <- 1
      x <- select(results, team, year, firstname, surname, PAV_total)
      print(x)
    }
    else{
      results <- fetch_squiggle_data("pav", year=year_num, team=team_id)
      y <- select(results, team, year, firstname, surname, PAV_total)
      print(y)
      x <- rbind(x,y)
    }
  }
}
write.csv(x, "/home/chris/Documents/Mitch/AFL_Data/AFL_Data/django_AFL_ML/R_Code/all_PAVs.csv", row.names=FALSE)
