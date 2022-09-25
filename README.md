# AFL_Data

Read about the project here: https://mitchgill16.github.io/AFL_Data/

How to run the project.

1. Clone the repository
2. cd into the main folder (AFL_ML)
3. Install the python packages required in a conda environment from requirements.txt

      .a This might take a little bit as since I've been experimenting with a lot of random things during the development and this casued the packages to build up
4. Make sure R is installed and the tidyverse packages have been installed.

      .a along with the fitzRoy R package available here: https://github.com/jimmyday12/fitzRoy
  
5. Predict the upcoming round after teams have been announced (usually on thursday at 6:40pm AEST) by running the following command in the main folder
  
  bash run_predict.sh input_year input_round 
  
  NOTE: rounds 1-23 are straight forward.
  Finals week 1 (Qualifying and Elimination Finals) = 24
  Finals week 2 (Semi Finals) = 25
  Finals week 3 (Preliminary Finals) = 26
  Finals week 4 (Grand Final) = 27
  eg. I am writing this readme in the couple days before the 2022 AFL Grand Final so I would type
  
  bash run_predict.sh 2022 27
    
6. Once The round is over update the match information and data. This step is necessary as the models work by using previous games data to make their predictions. Use the run_data_update.sh script to achieve this
 
 bash run_data_update.sh input_year input_round
  
I intend to intend to continue updating the matches into 2023, so you can always skip step 6 and just reclone the repository whenever you want to make
a prediction 
