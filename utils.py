import numpy as np
import math
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def truncate_and_pad(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        padding = [0.0] * (max_length - len(sequence))
        return sequence + padding
    
    

def getNormSeries(df):
    df['norm'] = df['norm'].apply(lambda seq: truncate_and_pad(seq, max_length=128))
    
    # Create one norm list by unique label

    norms_by_label = {}

    # Iterate through the DataFrame and organize norms by label
    for index, row in df.iterrows():
        label = row['label']
        norm_values = row['norm']
        
        if label not in norms_by_label:
            norms_by_label[label] = []
        
        norms_by_label[label].extend(norm_values)

    # Create a list of dictionaries for the DataFrame
    data_list = [{'label': label, 'norm': norm_values} for label, norm_values in norms_by_label.items()]

    # Create a pandas DataFrame from the list of dictionaries
    grouped_df = pd.DataFrame(data_list)
    
    # Create Darts series
    series_list = []
    scaler = Scaler()

    for label, norm_list in norms_by_label.items():
        series = TimeSeries.from_values(np.array(norm_list))
        # series_list.append(series)
        
        # Normalize the time series
        series_scaled = scaler.fit_transform(series)
        series_list.append(series_scaled)

    # Add the series_list to the new_df DataFrame
    grouped_df['norm_series'] = series_list
    
    # Create a dictionaries
    series_dict = {
        "walk" : grouped_df['norm_series'][0][:1024],
        "rest" : grouped_df['norm_series'][1][:1024],
        "run" : grouped_df['norm_series'][2][:1024],
        "tackle" : grouped_df['norm_series'][3][:1024],
        "dribble" : grouped_df['norm_series'][4][:1024],
        "pass" : grouped_df['norm_series'][5][:1024],
        "cross" : grouped_df['norm_series'][6][:1024],
        "shot" : grouped_df['norm_series'][7][:1024],
        "no action" : grouped_df['norm_series'][8][:1024],
    }
    
    return series_dict, scaler
     
    
