install libraries in ./requirements.txt

To run the app use the command 'streamlit run ./app"
(Change matches data paths if needed)

here is an attempt to solve the problem using Darts library which allows to work on time series and mutlivariate series (the result are not very conclusive):
  - generating a game can be slow depending on your machine, start by generate a game of 1 minute
  - offensive/defensive style aren't implemented
  - randomness between several games is not implemented 
  - the model for labels generate too much "no action" and "passe" but not enough "dribble"
  - the model norm values is hard to set up in order to fit all labels (using 1 model for each label might work better)
  - the gait generated always have a length of 128 (it's because of the padding which should be remove if the model was better)
  
