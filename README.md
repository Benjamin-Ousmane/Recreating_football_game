install libraries in ./requirements.txt

To run the app use the command 'streamlit run ./app"
(Change matches data paths if needed)

The algorithm is a prototype using Transformers for timeseries (I wanted to see what's possible with the Darts library) :
  - generating a game can be slow depending on your machine, start by generate a game of 1 minute
  - offensive/defensive style aren't implemented
  - randomness between several games is not implemented 
  - the model generate to much "no action" and "passe" but not enough "dribble"
