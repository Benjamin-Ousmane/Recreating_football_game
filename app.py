
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from darts import TimeSeries
from darts.models import NBEATSModel, TransformerModel
import gc

from sklearn.preprocessing import LabelEncoder

from random import seed
from random import randint
 
from utils import getNormSeries

@st.cache_data
def load_model(url):
    model_loaded = NBEATSModel.load(url)
    return model_loaded

@st.cache_data
def predict_norms(n_predictions, _model):
    norm_pred = _model.predict(n_predictions)
    return norm_pred



gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.is_available()

st.set_page_config(layout="centered")

st.title("Generative AI recreating football game")

### 1. -------------------------------------------------------------------------------------------------
st.header("1. Simple data analysis")

# load data (change paths of needed)
df1 = pd.read_json("./match_1.json")  
df2 = pd.read_json("./match_2.json")
df = pd.concat([df1, df2])
df = df.reset_index(drop=True)
df
## 1.1.
st.subheader("1.1. Analysis of gait length")

# calculate gait length 
df['norm_length'] = df['norm'].apply(len)

# describe gait length group by labels
df_stats_len_action = df.groupby('label').describe()
df_stats_len_action[:1] = df_stats_len_action[:1].astype(str)
st.write(df_stats_len_action)
st.write(
    '''
        Comments :
        - The player mostly runs or walks, then dribbles (as expected) 
        - Walk and cross gaits take on average longer than other actions (without counting rest)
    ''')

## 1.2.
st.subheader("1.2. Analysis of the acceleration norm") 

st.write("Here's what the player acceleration norm looks like over a 2-minute period : ")

df_exploded = df.explode('norm')
df_exploded = df_exploded.reset_index()
df_exploded['gait'] = df_exploded['index']
df_exploded['time'] = df_exploded.index * 0.02

# reorder columns
cols = ["time", "gait", "label", "norm"]
df_exploded = df_exploded[cols] 

window_size = 120 # 2 minutes

# Create a Streamlit slider for controlling the sliding window
window_start = st.slider('Select Window Start (seconds)', 0.0, df_exploded['time'].values[-1]-window_size, 0.0, step=0.02)

# Define the window range based on the slider value
window_end = window_start + window_size

# Filter the DataFrame based on the window range
@st.cache_data
def filter_dataframe(window_start, window_end, df):
    df_filtered = df[(window_start <= df['time']) & (df['time']<= window_end)]
    return df_filtered

# Get the filtered DataFrame using the slider values
df_window = filter_dataframe(window_start, window_end, df_exploded)

# Create an Altair chart
chart_game_norm = alt.Chart(df_window).mark_line(opacity=0.8, size=1).encode(
    x=alt.X('time', title='Time (seconds)'),
    y='norm',
    color='label',
    detail='gait'
).properties(
    width=600,
    height=400
)

# Display the chart in Streamlit
st.altair_chart(chart_game_norm, theme=None, use_container_width=True)


st.write("We can also observe more precisely the norm during one gait with the graph above :")
# Select boxes for label and number of lines to display
col1, col2 = st.columns([3,1])
with col1:
    selected_label = st.multiselect("Select Label(s)", df['label'].unique(), default=["walk"])
with col2:
    num_lines = st.selectbox("Number of Lines by label", [1, 3, 5, 10, 20, 50])

if selected_label!= []:
    df_filtered = df.groupby('label').head(num_lines)
    df_filtered = df_filtered[df_filtered["label"].isin(selected_label)]
    
    # Create a list of dictionaries for each label and gait
    line_data = []
    for index, row in df_filtered.iterrows():
        label = row['label']
        norm_values = row['norm']
        for i, norm in enumerate(norm_values):
            line_data.append({'gait': index, 'label': label, 'norm': norm, 'x': i*0.02}) # 50Hz -> 1 row every 0.02s
            
    # Convert the list of dictionaries back to a DataFrame
    line_df = pd.DataFrame(line_data)

    # Create an Altair multiline chart
    chart_gait_norm = alt.Chart(line_df).mark_line(opacity= 0.8, size=1).encode(
        x=alt.X('x', title='Time (seconds)'),
        y='norm',
        color='label',
        detail='gait'
    ).properties(
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.altair_chart(chart_gait_norm, theme=None, use_container_width=True)
st.write(
'''
    Comments :
    - Rest, Walk, Run, Dribble can last several gaits in a row
    - Pass, Shot, Cross, Tackle usually last for only one gait 

''')

### 2. ------------------------------------------------------------------------------------------------
st.header("2. List of different approaches") 
st.write(
    '''
        - Generate a sequence of labels, then generate a list of norms for each actions
        - Generate a long sequence of norms, then classify this sequence in sub sequences depending on labels
        - Generate labels and norms for each gait one by one
    ''')

### 3. ------------------------------------------------------------------------------------------------
st.header("3. Model") 
st.write("Description of the algorithms in the file Generative_AI_recreating_football_game.ipynb")



### 4. ------------------------------------------------------------------------------------------------
st.header("4. Recreating games") 

# preprocessing 
series_dict, norm_scaler = getNormSeries(df.copy())

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df["label"])

# load models 

label_model = TransformerModel.load("./models/label_model.pkl")
norm_model = TransformerModel.load("./models/norm_model.pkl")


if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []

# selectboxes for game length and game type
c1, c2, c3 = st.columns(3)
with c1:
    game_length = st.selectbox("Game length (minutes)", [1, 5, 15, 20, 60])
    game_length_sec = game_length * 60
with c2:
    game_type = st.selectbox("Game type", ["normal", "attacking", "defending"])
# button to generate games
with c3:
    st.write("")
    st.write("")
    generate_game = st.button("Generate game")
    

if generate_game :  
    game = []
    label_series = None


    while (game_length_sec > 0 ):
        # Generate a sequence of labels
        generated_labels = label_model.predict(n = 300, series=label_series) 
        generated_labels = generated_labels.map(lambda x: np.round(x))
        generated_labels = generated_labels.map(lambda x: np.clip(x, 0, 8))
        generated_labels_decoded = label_encoder.inverse_transform(generated_labels.values().astype(int))
        
        # Generate a sequence of norms for each label
        for label in generated_labels_decoded:
            generated_norm_scaled = norm_model.predict(n=128, series=series_dict[label])
            series_dict[label] = generated_norm_scaled# add some changes for the next gaits
            
            generated_norm = norm_scaler.inverse_transform(generated_norm_scaled).values()
             
            # Find the index of the first occurrence of 0
            # first_zero_index = np.argmax(generated_norm <= 0)

            # Cut the array at the first zero index
            # generated_norm = generated_norm[:first_zero_index].tolist()
            
            game.append({
                'norm':generated_norm,
                'label':label
            })
            
            st.write(generated_norm)

            
            game_length_sec = game_length_sec - len(generated_norm)*0.02
            if game_length_sec < 0 :
                break
        
        # Prepare  the next sequence of labels
        label_series = generated_labels
        
    st.session_state['predictions'] = game


if st.session_state['predictions'] != []:

    output_json = json.dumps(st.session_state['predictions'])

    st.json(output_json, expanded=True)

    st.download_button(
        label = "Download output as json",
        data = output_json,
        file_name='generated_game.json',
        mime="application/json",
    )