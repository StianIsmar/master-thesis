import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


import wt_data
import ff_transform
import data_statistics

def create_rms_datasets_for_ISO_frequencies(wt_instance):
    intervals = wt_instance.ten_second_intervals
    all_rms_data = []
    for i, interval in enumerate(intervals):
        if i > 0:
            break
        if (interval.sensor_df.shape[1]) == 14 and (interval.op_df["PwrAvg;kW"][0] > 0):
            print(f'Checking interval: {i} / {len(intervals)-1}', end='\r')
            rot_data = interval.high_speed_rot_data
            peak_array = interval.high_speed_peak_array
            avg_speed = rot_data['mean']
            avg_power = interval.op_df["PwrAvg;kW"][0]
            active_power = interval.op_df["PwrAct;kW"][0]
            wind_speed = interval.op_df["WdSpdAct;m/s"][0]
            nacelle_direction = interval.op_df["NacDirAct;deg"][0]
            time_stamps = interval.sensor_df['TimeStamp']

            interval_data = []
            interval_data.append(avg_power)
            interval_data.append(active_power)
            interval_data.append(wind_speed)
            interval_data.append(nacelle_direction)

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

                fft, time, centroid, rms = fast.fft_transform_time(plot=True)
                interval_data.append(rms)

            all_rms_data.append(interval_data)

    # Crete a dataframe for all rms_data
    df_column_names = ['AvgPower',
                       'ActPower',
                       'WindSpeed',
                       'NacelleDirection',
                       'GnDe_RMS',
                       'GnNDe_RMS',
                       'MnBrg_RMS',
                       'GbxRotBrg_RMS',
                       'Gbx1Ps_RMS',
                       'GbxHssFr_RMS',
                       'GbxHssRr_RMS',
                       'Gbx2Ps_RMS',
                       'GbxIss_RMS',
                       'NacZdir_RMS',
                       'NacXdir_RMS',
                     ]
    df = pd.DataFrame(all_rms_data, columns=df_column_names)

    return df



''' 
Creates a dataframe for each vibration measurement. Splits the frequency domain into different a set of bins,
calculates rms for each bin and stores it as a feature in the dataframe.
Input: 
wt_instance = a wind turbine instace (object)
sensor_name = name of the sensor to analyse (string)
bins = number of bins (int) default is 25
calc_rms_for_bins = calculates rms for each bin (boolean)
plot = plots the FFT transformation (boolean)
plot_vertical_lines = blots a separation line between each bin (boolean)
Output:
df = pandas dataframe with rms value for each bin
'''
def create_rms_datasets_for_one_component(wt_instance, sensor_name, power_threshold=0, bins=25,
                                          calc_rms_for_bins=True, plot=True,
                                          plot_vertical_lines=True, avg_pwr_values=[]):
    intervals = wt_instance.ten_second_intervals
    whole_dataset = []
    type = ''
    rms_bins = []

    if (sensor_name == 'TimeStamp') or (sensor_name == 'Speed Sensor;1;V') or (sensor_name == 'LssShf;1;V'):
        # Skip TimeStamp
        return
    elif sensor_name.find('Gbx'):
        # Run RMS Calculation on Gearbox vibration data
        type = 'gearbox'
    elif sensor_name.find('MnBrg'):
        # Run RMS Calculation on Main Bearing vibration data
        type = 'nacelle'
    elif sensor_name.find('Gn'):
        # Run RMS Calculation on Generator vibration data
        type = 'generator'
    elif sensor_name.find('Nac'):
        # Run RMS Calculation on Nacelle vibration data
        type = 'nacelle'

    counter = 0
    print_int = -1000
    for i, interval in enumerate(intervals):
        interval_data = []
        #if i > 50:
            #break

        if interval.op_df["PwrAvg;kW"][0] in avg_pwr_values:
            print_int = i
            print(print_int)

        # We only want to use data which has measurement for all signals and positive average power
        if (interval.sensor_df.shape[1]) == 14 and (interval.op_df["PwrAvg;kW"][0] > power_threshold):

            if (len(avg_pwr_values) != 0) and (i != print_int):
                continue


            print(f'Checking interval: {i} / {len(intervals)-1}', end='\r')
            counter += 1
            avg_power = interval.op_df["PwrAvg;kW"][0]
            rot_data = interval.high_speed_rot_data
            active_power = interval.op_df["PwrAct;kW"][0]
            wind_speed = interval.op_df["WdSpdAct;m/s"][0]
            nacelle_direction = interval.op_df["NacDirAct;deg"][0]
            time_stamps = interval.sensor_df['TimeStamp']

            interval_data.append(avg_power)
            interval_data.append(active_power)
            interval_data.append(rot_data['mean'])
            interval_data.append(wind_speed)
            interval_data.append(nacelle_direction)

            vibration_signal = interval.sensor_df[sensor_name]
            title_name = sensor_name.split(';')[0]
            fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, type)
            fft, time, centroid, rms, rms_bins, x = fast.fft_transform_time(rot_data,
                                                                         avg_power,
                                                                         name=title_name,
                                                                         calc_rms_for_bins=calc_rms_for_bins,
                                                                         plot=plot,
                                                                         bins=bins,
                                                                         plot_vertical_lines=plot_vertical_lines)
            # Add each rms value for all bins into interval_data
            for i, rms_val in enumerate(rms_bins):
                interval_data.append(rms_val)

            whole_dataset.append(interval_data)

    df_column_names = ['AvgPower', 'ActPower', 'AvgRotSpeed', 'WindSpeed', 'NacelleDirection']
    for i in range(len(rms_bins)):
        signal_rms_name = f"{sensor_name.split(';')[0]}_RMS_{i}"
        df_column_names.append(signal_rms_name)

    # Crete a dataframe for all rms_data
    print(f'{counter} / {len(intervals)-1} intervals added to dataframe')
    df = pd.DataFrame(whole_dataset, columns=df_column_names)
    return df


def train_test_split(df, percentage):
    split_index = int(np.floor(df.shape[0]) * percentage)
    train = df[:split_index]
    test = df[split_index:].reset_index(drop=True)
    return train, test


def save_dataframe_pickle(dataframe, name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    pickle.dump(dataframe, open(path + name + '.p', 'wb'))
    print(f'Saved {name}.')

def load_dataframe_pickle(name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    dataframe = pickle.load(open(path + name + '.p', 'rb'))
    print(f'Loaded {name}')
    return dataframe


def save_dataframe_to_csv(df, name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    df.to_csv(path + name, index=False)
    print(f'Saved {name}.')


def load_dataframe_from_csv(name):
    path = '/Volumes/OsvikExtra/VibrationData/RMS_dataset/'
    df = pd.read_csv(path + name)
    print(f'Loaded {name}')
    return df


def plot_column(df):
    x_values = np.arange(0, df.shape[0])
    for i, col_name in enumerate(df.columns.values):
        plt.figure(figsize=(15, 8))
        sns.lineplot(x_values, df[col_name])
        plt.title(f'Plot of {col_name}')
        plt.xlabel('Interval')
        plt.ylabel(col_name)
        plt.margins(0)
        plt.show()

#wt_instance = wt_data.create_wt_data('WTG01', save_minimal=False)
#wt_instance = wt_data.load_instance("WTG01",load_minimal=False)
#df = create_rms_datasets_for_one_component(wt_instance, 'GnDe;0,0102;m/s2', power_threshold=2500,
#                                           plot=True, bins=50, plot_vertical_lines=True)


# GENERATOR 'GnDe;0,0102;m/s2'
# GEARBOX 'GbxHssRr;0,0102;m/s2'

def save_all_df_for_component(component_name,bins,power_threshold=2500):

    for i in range(3):
        i = i + 2
        turbine_name = f"WTG0{i}"
        wt_instance = wt_data.load_instance(turbine_name, load_minimal=False)
        df = create_rms_datasets_for_one_component(wt_instance, component_name, power_threshold=2500,
                                                   plot=False, bins=bins, plot_vertical_lines=False)
        save_dataframe_to_csv(df, f'{turbine_name}_{component_name}_RMS_power>2500.csv')
        del wt_instance

#save_all_df_for_component('GbxHssRr;0,0102;m/s2',bins=50,power_threshold=2500)





#df = load_dataframe('WTG01_RMS')
#train, test = train_test_split(df, 0.8)


#plot_column(train)
#train.hist()
#data_statistics.plot_histograms(train)

#data_statistics.boxplot_rms(train, name='Training Set')
#data_statistics.boxplot_rms(test, name='Testing Set')