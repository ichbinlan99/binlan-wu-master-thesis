import os
import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np


#todo move path outside
def load_data(user, sub_folder):
    path = '/Users/binlanwu/Downloads/TUM/5.Master_Thesis_eth/graphical_modeling/data/ecg-bp-' +str(user) + '/' + str(sub_folder) + '/'
    data = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".csv.gz"):
                file_path = os.path.join(root, file_name)
                data.append(pd.read_csv(file_path, compression ='gzip'))
    return pd.concat(data)

# def load_data(user, sub_folder):
#     path = '/Users/binlanwu/Downloads/TUM/5.Master_Thesis(ETH)/graphical_modeling/data/ecg-bp-' +str(user) + '/' + str(sub_folder) + '/'
#     data = []
#     for root, dirs, files in os.walk(path):
#         for file_name in files:
#             if file_name.endswith(".csv.gz"):
#                 file_path = os.path.join(root, file_name)
#                 data.append(pd.read_csv(file_path, compression='gzip'))
#     return pd.concat(data)


def to_unix_time(time_str: str):
    """Converts a datetime string to a Unix timestamp.

    Args:
    time_str (str): A datetime string in the format "%Y-%m-%d %H:%M:%S".

    Returns:
        float: The Unix timestamp of the input datetime string.

    Raises:
        ValueError: If the input datetime string is not in the correct format.
    """
    try:
        time_to_convert = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        converted_time = datetime.datetime.timestamp(time_to_convert)
        return converted_time
    except ValueError:
        print('Baseline start and end have not been defined properly!')
        return

def cut_signal(df_to_cut: pd.DataFrame, start_time: float, end_time: float):
    """Cut a pandas DataFrame between specified start and end time.

    Args:
        df_to_cut (pd.DataFrame): The DataFrame to be cut.
        start_time (float): The start time in format "%Y-%m-%d %H:%M:%S".
        end_time (float): The end time in format "%Y-%m-%d %H:%M:%S".

    Returns:
        pd.DataFrame: The cut DataFrame.
    """
    mask_cors = (df_to_cut.index > start_time) & (df_to_cut.index < end_time)
    df_cut = df_to_cut.loc[mask_cors]
    return df_cut

def get_data_path():
    return '/Users/binlanwu/Downloads/TUM/5.Master_Thesis_eth/graphical_modeling/data/src'

def load_user_table():
    measurement_intervals = pd.read_csv('data_preprocessing/MeasurementMetadata.csv').set_index('user_id')
    resting_table = pd.read_csv('data_preprocessing/UserHRTResting.csv').set_index('user_id')
    return measurement_intervals, resting_table

def load_time_df_baseline(df, measurement_intervals, user):
    # print(measurement_intervals)
    time_data: dict[str | Any, float | Any] = {
        'start_time': measurement_intervals.loc[user, 'start_time'],
        'end_time': measurement_intervals.loc[user, 'end_time'],
        'baseline_start': measurement_intervals.loc[user, 'baseline_start'],
        'baseline_end': measurement_intervals.loc[user, 'baseline_end']
    }
    for key, value in time_data.items():
        time_data[key] = to_unix_time(value)
    df_baseline = cut_signal(df, time_data['baseline_start'], time_data['baseline_end'])
    df = cut_signal(df, time_data['start_time'], time_data['end_time'])
    return df, df_baseline, time_data

def robust_normalize(data):
    median = np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    normalized_data = (data - median) / iqr
    return normalized_data