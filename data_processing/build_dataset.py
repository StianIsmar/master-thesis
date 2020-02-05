import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import wt_data
import ff_transform


def create_rms_datasets(wt_instance):
    intervals = wt_instance.ten_second_intervals
    all_rms_data = []
    for i, interval in enumerate(intervals):
        #if i > 40:
            #break
        if (interval.sensor_df.shape[1]) == 14 and (interval.op_df["PwrAvg;kW"][0] > 0):
            print(f'Checking interval: {i} / {len(intervals)}', end='\r')
            rot_data = interval.high_speed_rot_data
            peak_array = interval.high_speed_peak_array
            avg_speed = rot_data['mean']
            avg_power = interval.op_df["PwrAvg;kW"][0]
            time_stamps = interval.sensor_df['TimeStamp']

            interval_rms_data = []
            interval_rms_data.append(avg_power)
            for i, column_name in enumerate(interval.sensor_df.columns.values):
                vibration_signal = interval.sensor_df[column_name]
                if (column_name == 'TimeStamp') or (column_name == 'Speed Sensor;1;V') or (column_name == 'LssShf;1;V'):
                    # Skip TimeStamp
                    continue
                elif column_name.find('Gbx'):
                    # Run RMS Calculation on Gearbox vibration data
                    fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, 'gearbox')
                elif column_name.find('MnBrg'):
                    # Run RMS Calculation on Main Bearing vibration data
                    fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, 'nacelle')
                elif column_name.find('Gn'):
                    # Run RMS Calculation on Generator vibration data
                    fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, 'generator')
                elif column_name.find('Nac'):
                    # Run RMS Calculation on Nacelle vibration data
                    fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, 'nacelle')

                fft, time, centroid, rms = fast.fft_transform_time()
                interval_rms_data.append(rms)

            all_rms_data.append(interval_rms_data)

    # Crete a dataframe for all rms_data
    df = pd.DataFrame(all_rms_data, columns=['AvgPower', 'GnDe_RMS', 'GnNDe_RMS', 'MnBrg_RMS', 'GbxRotBrg_RMS',
                                             'Gbx1Ps_RMS','GbxHssFr_RMS','GbxHssRr_RMS', 'Gbx2Ps_RMS', 'GbxIss_RMS',
                                             'NacZdir_RMS', 'NacXdir_RMS'])
    return df

def train_test_split(dataframe, percentage):
    split_index = int(np.floor(df.shape[0]) * percentage)
    train = dataframe[:split_index]
    test = dataframe[split_index:].reset_index(drop=True)
    return train, test


def save_dataframe(dataframe, name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    pickle.dump(dataframe, open(path + name + '.p', 'wb'))
    print(f'Saved {name}.')

def load_dataframe(name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    dataframe = pickle.load(open(path + name + '.p', 'rb'))
    print(f'Loaded dataframe.')
    return dataframe

def plot_column(df):
    x_values = np.arange(0, df.shape[0])
    for i, col_name in enumerate(df.columns.values):
        plt.figure(figsize=(15, 8))
        plt.plot(x_values, df[col_name])
        plt.title(f'Plot of {col_name}')
        plt.xlabel('Interval')
        plt.ylabel(col_name)
        plt.margins(0)
        plt.show()





#wt_instance = wt_data.load_instance("WTG01",load_minimal=False)
#df = create_rms_datasets(wt_instance)
df = load_dataframe('WTG01_RMS')
train, test = train_test_split(df, 0.8)

plot_column(train)