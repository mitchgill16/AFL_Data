**Predicting The 2022 AFL Season with Machine Learning**

<br />

# Contents
{:.no_toc}
* Data 
{:toc}

<br />

## Summary
    • What started out as a way to practice python during early COVID-19 for my masters has become a fun hobby to demonstrate skills I’ve learnt over the past few years 
    
    • Used data from a variety of sources to assemble a large training testing dataset
    
    • Trained up 3 xgboost models based on n previous games worth of data (where n = 2,3,10) and the upcoming round information such as players
    
    • Updated the dataset round by round to then predict the upcoming round for each round of the 2022 season
    
    • Finished with xyz tips and ranked xyz overall on ESPN tipping
    
    • Code and combined dataset available
    
    • contact me at mitchgill16@gmail.com or [personal website]()  which has other projects I’ve worked on.
    
    • Special thanks to my fiancee Rebecca who has listened to me ramble about this project over the last couple years and has heard the phrase ‘I’m just gonna quickly update the model’ or ‘I’m just running the predictions’ more times than anyone should hear. Additional thanks to Philipp Bayer, who is someone I look up to and was also my supervisor whilst I was at uni. During this time he helped my develop my skills for my university projects which has transferred over into to making this personal project possible. Check out his twitter for the cool work he’s doing at the Minderoo foundation.

## Background

    • At the start of 2020 knew I was going to be working on a ML project with the Applied Bioinformatics group which was based on predicting soybean traits from genetic information (Recently published [here](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z) :D )

    • COVID happened so I wanted to practice + upskill my python skills whilst learning how to do something with ML
    
    • I like the footy and I like what [squiggle does](https://squiggle.com.au/)
    
    • So I looked up a tutorial on what ML is, did coursera’s DL classes & made plenty of mistakes and dumb decisions which would be fixed a long the way.

## Acquiring the Data & Making the Dataset

    • Naive webscraper which scraped every match from 2011 onwards and stored them in team spreadsheets for most stats available on the website
    
    • FitzRoy package to acquire the venue, ladder and lineup information for each of these games
    
    • HPN Player Approximate Value (PAV) to determine a rough value system for the lineups for each team. 
      
## Training the Models

    • Assembled the datasets for n previous games with this data and what would be possible known about the upcoming game such as venue and PAV.
    
    • After a lot of experimentation, hyperparamter tuning and trial + error optimisation decided that the best combination was to get each of the 3 models to make a prediction in a best of 3 approach. Latest run had this at approx 68% accuracy and about 3%SD for the stratified k-fold testing
    
    • Also trained a regression model at the same time to get a margin prediction
      
## Round by Round Prediction of the 2022 Season

    • All well and good that could predict games results for games that had already happened. 
    
    • Made a way to use fitzRoy to get upcoming game information and lineups (when they’re announced) assign a PAV value and then to make the equivalent amount of previous games information that can work with the model
    
    • So for each round I would update the dataset with the previous rounds information, wait for teams to be announced and then predict the upcoming round. 
    
    • I’d then set my tips as the models best of 3 prediction along with margin
    
    • Overall worked very well for this season although it outperformed the training testing accuracy which suggests that this season did not have a lot of upsets round to round. Additionally it suggests that the model did get quite lucky with 50/50 games & some upsets (For example it tipped Gold Coast to beat Richmond when most tippers picked Richmond, and fortunately Noah Anderson kicked a goal after the siren). Close games are usually 50/50, unless you’re Collingwood, but they’ll regress to the mean eventually. 
      
## Code & Conclusion

    • Code is available to download and play around with or if you want to use the data I’ve collected.
    
    • Overall I’m pretty happy with this project.
    
    • I intend to update the data round by round and do predictions for the 2023 season.
    
    • I would like to try and fix the webscraper for the footywire data as this was something I made in April 2020, and after 2-3 years more experience with python I think I could improve the efficiency of the system that is in place. 
