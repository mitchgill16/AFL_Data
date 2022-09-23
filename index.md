**Predicting The 2022 AFL Season with Machine Learning**

<br />

# Contents
{:.no_toc}
* Data 
{:toc}

<br />

## Summary
Using data from a variety of dedicated AFL stats datasets I was able to assemble a test and training dataset to train XGBoost models to predict the 2022 season round by round. This method ended up tipping xyz tips correct and ranking me overall xyz for the ESPN footy tips competition. The XGBoost models were trained on match stats for the previous games with a model learning from the previous 2 games, another model from the previous 3 games and another learning from the previous 10 games. The models, code and combined dataset are available in the repository for this project.
    
Feel free to contact me at mitchgill16@gmail.com with any questions or view my [personal website]() (currently under development)  to check out other projects I've worked on.
    
Special thanks to my fiancee Rebecca who has listened to me ramble about this project over the last couple years and has heard the phrase ‘I’m just gonna quickly update the model’ or ‘I’m just running the predictions’ more times than anyone should hear. Additional thanks to Philipp Bayer, who is someone I look up to and was also my supervisor whilst I was at uni. During this time he helped my develop my skills for my university projects which has transferred over into to making this personal project possible. Check out his [twitter](https://twitter.com/PhilippBayer) for the cool work he’s doing at the Minderoo foundation.

## Background
At the start of 2020 knew I was going to be working on a machine learning (ML) project with the Applied Bioinformatics group which was based around predicting soybean traits from genetic information (Recently published [here](https://bmcplantbiol.biomedcentral.com/articles/10.1186/s12870-022-03559-z) :D ). Then COVID happened so I wanted to practice + upskill my python skills whilst learning how to do something with ML during lockdowns and restrictions. I'm a diehard Freo supporter and will probably watch most other games on. I also really like what [squiggle does](https://squiggle.com.au/) along with others in this space. So I got to work learning how to implement ML in python, did coursera’s deep learning classes & made plenty of mistakes and questionable design choices which would be fixed and changed during the course of a couple years of edits and practice.

## Acquiring the Data & Making the Dataset
The step in this project was to gather data sources for each game that has been played, to build a dataset. I decided to build (a fairly naive) webscraper which scraped every match from 2013 onwards (as both expansion teams have had been in the competition for at least 1 year by this point), from the [footywire website](https://www.footywire.com/) and stored them in team spreadsheets for both basic stats and advanced stats available on the website. I later added in data from the FitzRoy package to acquire the venue, ladder and lineup information for each of these games. And finally to account for the impact of individual players I added in a team aggregated Player Approximate Value (PAV) from [HPN footy](https://www.hpnfooty.com/) to determine a rough value system for each of the lineups for each team for each match.

Once the dataset had been acquired for each game and allocated to each team, it was then time to assmeble the train/test dataset. The project works by using n previous games worth of information, along with data that is possible to know about the upcoming games. 

**TO ADD: Diagram with logos of data sources -> into excel -> n games spreadsheet>**
      
## Training the Models
Assembled the datasets for n previous games with this data and what would be possible known about the upcoming game such as venue and PAV. After a lot of experimentation, hyperparamter tuning and trial + error optimisation decided that the best combination was to get each of the 3 models to make a prediction in a best of 3 approach. Latest run had this at approx 68% accuracy and about 3%SD for the stratified k-fold testing. Also trained a regression model at the same time to get a margin prediction
      
## Round by Round Prediction of the 2022 Season
All well and good that could predict games results for games that had already happened. Made a way to use fitzRoy to get upcoming game information and lineups (when they’re announced) assign a PAV value and then to make the equivalent amount of previous games information that can work with the model. So for each round I would update the dataset with the previous rounds information, wait for teams to be announced and then predict the upcoming round. I’d then set my tips as the models best of 3 prediction along with margin
    
Overall worked very well for this season although it outperformed the training testing accuracy which suggests that this season did not have a lot of upsets round to round. Additionally it suggests that the model did get quite lucky with 50/50 games & some upsets (For example it tipped Gold Coast to beat Richmond when most tippers picked Richmond, and fortunately Noah Anderson kicked a goal after the siren). Close games are usually 50/50, unless you’re Collingwood, but they’ll regress to the mean eventually. 
      
## Code & Conclusion
Code is available to download and play around with or if you want to use the data I’ve collected. Overall I’m pretty happy with this project. I intend to update the data round by round and do predictions for the 2023 season. I would like to try and fix the webscraper for the footywire data as this was something I made in April 2020, and after 2-3 years more experience with python I think I could improve the efficiency of the system that is in place. 
