import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


import wt_data
import resample
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
def create_rms_datasets_for_one_component(wt_instance,
                                          sensor_name,
                                          power_threshold=0,
                                          rpm_threshold=0,
                                          resample_signal=False,
                                          spectrum_type='time',
                                          interpolation_method='linear',
                                          bins=25,
                                          get_rms_for_bins=True,
                                          plot=True,
                                          plot_bin_lines=True,
                                          x_lim=None,
                                          frequency_lines=[],
                                          horisontal_lines=[],
                                          avg_pwr_values=[],
                                          interval_range=[],
                                          number_of_resample_points=1500,
                                          plot_resampling=False,
                                          round_plots=1,
                                          high_speed=True):

    intervals = wt_instance.ten_second_intervals
    whole_dataset = []
    type = ''
    rms_bins = []
    title_name = f"{wt_instance.name} {sensor_name.split(';')[0]}"
    fast=None

    if (spectrum_type != 'time') and (spectrum_type != 'order'):
        print('Input variable spectrum_type must either be "order" or "time"! \n')
        return None

    print(f"This is sensor name {sensor_name}")

    if (sensor_name == 'TimeStamp') or (sensor_name == 'Speed Sensor;1;V') or (sensor_name == 'LssShf;1;V'):
        # Do nothing for TimeStamp, SpeedSensor or LssShft
        return
    elif sensor_name.find('Gbx') != -1:
        # Run RMS Calculation on Gearbox vibration data
        type = 'gearbox'
    elif sensor_name.find('MnBrg') != -1:
        # Run RMS Calculation on Main Bearing vibration data
        type = 'nacelle' # same as bearing
    elif sensor_name.find('Gn') != -1:
        # Run RMS Calculation on Generator vibration data
        type = 'generator'
    elif sensor_name.find('Nac') != -1:
        # Run RMS Calculation on Nacelle vibration data
        type = 'nacelle'
    print(f"Type of component registered when building dataset is {type}.\n")
    counter = 0
    print_int = -1000
    for i, interval in enumerate(intervals):
        interval_data = []

        avg_power = interval.op_df["PwrAvg;kW"][0]
        rot_data = interval.high_speed_rot_data
        mean_rpm = rot_data['mean']

        # We only want to use data which has measurement for all signals and positive average power
        if (interval.sensor_df.shape[1]) == 14 and (avg_power > power_threshold) and (mean_rpm > rpm_threshold) and \
                ((interval.op_df["PwrAvg;kW"][0] in avg_pwr_values) or (i in interval_range)):


            print(f'Checking interval: {i} / {len(intervals)-1}', end='\r')
            counter += 1
            avg_power = interval.op_df["PwrAvg;kW"][0]
            if high_speed:
                rot_data = interval.high_speed_rot_data
                peak_array = interval.high_speed_peak_array
            else:
                rot_data = interval.low_speed_rot_data
                peak_array = interval.low_speed_peak_array
            active_power = interval.op_df["PwrAct;kW"][0]
            wind_speed = interval.op_df["WdSpdAct;m/s"][0]
            nacelle_direction = interval.op_df["NacDirAct;deg"][0]
            time_stamps = interval.sensor_df['TimeStamp']


            interval_data.append(avg_power)
            interval_data.append(active_power)
            interval_data.append(mean_rpm)
            # interval_data.append(x)
            interval_data.append(wind_speed)
            interval_data.append(nacelle_direction)

            vibration_signal = interval.sensor_df[sensor_name]
            if resample_signal:
                if interpolation_method == 'linear':
                    x_resampled, y_resampled, all_x_round_domain, all_y_resampled, all_x_time_domain = resample.linear_interpolation_resampling(
                        time_stamps,
                        vibration_signal,
                        peak_array,
                        number_of_resample_points=number_of_resample_points,
                        round_plots=round_plots,
                        plotting=plot_resampling,
                        printing=False,
                        name=f'     {title_name}     Interval: {i}',
                        interpolation_method=interpolation_method)
                    
                    if spectrum_type == 'time':
                        fast = ff_transform.FastFourierTransform(all_y_resampled, all_x_time_domain, type)
                    elif spectrum_type == 'order':
                        fast = ff_transform.FastFourierTransform(all_y_resampled, all_x_round_domain, type)
                    else:
                        print('spectrum_type must be "time" or "order"')
                        return

                    fft, time, centroid, rms_bins, x = fast.fft_transform_order(rot_data,
                                                                                avg_power,
                                                                                name=title_name,
                                                                                interval_num=i,
                                                                                get_rms_for_bins=get_rms_for_bins,
                                                                                plot=plot,
                                                                                bins=bins,
                                                                                plot_bin_lines=plot_bin_lines,
                                                                                x_lim=x_lim,
                                                                                order_lines=frequency_lines,
                                                                                horisontal_lines=horisontal_lines,
                                                                                interpolation_method=interpolation_method
                                                                                )
                elif interpolation_method == 'cubic':
                    time_resampled, y_resampled, all_x_round_domain, all_y_resampled, all_x_time_domain = resample.cubic_interpolation_resampling(
                        time_stamps,
                        vibration_signal,
                        peak_array,
                        number_of_resample_points=number_of_resample_points,
                        round_plots=round_plots,
                        plotting=plot_resampling,
                        printing=False,
                        name=f'     {title_name}     Interval: {i}',
                        interpolation_method=interpolation_method)
                    
                    if spectrum_type == 'time':
                        fast = ff_transform.FastFourierTransform(all_y_resampled, all_x_time_domain, type)
                    elif spectrum_type == 'order':
                        fast = ff_transform.FastFourierTransform(all_y_resampled, all_x_round_domain, type)
                    else:
                        print('spectrum_type must be "time" or "order"')
                        return

                    fft, time, centroid, rms_bins, x = fast.fft_transform_order(rot_data,
                                                                                avg_power,
                                                                                name=title_name,
                                                                                interval_num=i,
                                                                                get_rms_for_bins=get_rms_for_bins,
                                                                                plot=plot,
                                                                                bins=bins,
                                                                                plot_bin_lines=plot_bin_lines,
                                                                                x_lim=x_lim,
                                                                                order_lines=frequency_lines,
                                                                                horisontal_lines=horisontal_lines,
                                                                                interpolation_method=interpolation_method
                                                                                )

            else:
                fast = ff_transform.FastFourierTransform(vibration_signal, time_stamps, type)
                fft, time, centroid, rms, rms_bins, x = fast.fft_transform_time(rot_data,
                                                                                avg_power,
                                                                                name=title_name,
                                                                                interval_num=i,
                                                                                plot=plot,
                                                                                get_rms_for_bins=get_rms_for_bins,
                                                                                bins=bins,
                                                                                plot_bin_lines=plot_bin_lines,
                                                                                x_max=x_lim,
                                                                                frequency_lines=frequency_lines,
                                                                                horisontal_lines=horisontal_lines
                                                                                )
            
            # Add each rms value for all bins into interval_data
            for i, rms_val in enumerate(rms_bins):
                interval_data.append(rms_val)

            whole_dataset.append(interval_data)

            # ---- TO FIGURE OUT THE FREQUENCY RANGE AROUND BIN 5 -------
            #print(f'Bin 5 is centered at {x[5]}. Delta is {x[1] - x[0]}. It spans the frequencies from {x[5] - (x[1] - x[0])} to {x[5] + (x[1] - x[0])}')

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
                                                   plot=False, bins=bins, plot_bin_lines=False)
        save_dataframe_to_csv(df, f'{turbine_name}_{component_name}_RMS_power>2500.csv')
        del wt_instance

#save_all_df_for_component('GbxHssRr;0,0102;m/s2',bins=50,power_threshold=2500)


def load_wt_high_freq_analysis(wt_name,component_name):
    wt_instance = wt_data.load_instance(wt_name, load_minimal=False)
    intervals_vibrations = []
    intervals_times = []
    intervals_data = []
    intervals_peak_arrays = []
    number_of_intervals = len(wt_instance.ten_second_intervals)
    
    for i, interval in enumerate(wt_instance.ten_second_intervals):
        if (interval.sensor_df.shape[1]) == 14 and (interval.op_df["PwrAvg;kW"][0] > 0):
            rot_data = interval.high_speed_rot_data

            avg_speed = rot_data['mean']
            avg_power = interval.op_df["PwrAvg;kW"][0]
            active_power = interval.op_df["PwrAct;kW"][0]
            wind_speed = interval.op_df["WdSpdAct;m/s"][0]
            nacelle_direction = interval.op_df["NacDirAct;deg"][0]

            
            interval_data = []
            interval_data.append(avg_power)
            interval_data.append(active_power)
            interval_data.append(wind_speed)
            interval_data.append(nacelle_direction)
            interval_data.append(avg_speed)
            
            # Peak array
            peak_array = interval.high_speed_peak_array
            intervals_peak_arrays.append(peak_array)

            
            # Vibration signal and timestamps
            vibration_signal = interval.sensor_df[component_name]
            time_stamps = interval.sensor_df['TimeStamp']
            time = interval.sensor_df['TimeStamp']

            intervals_vibrations.append(vibration_signal)
            intervals_times.append(time)
            intervals_data.append(interval_data)
            
    df_column_names = ['AvgPower',
                       'ActPower',
                       'WindSpeed',
                       'NacelleDirection',
                       'AvgSpeed']
    
    df_intervals_data = pd.DataFrame(intervals_data, columns=df_column_names)
            
    return intervals_vibrations, intervals_times, df_intervals_data, intervals_peak_arrays


#df = load_dataframe('WTG01_RMS')
#train, test = train_test_split(df, 0.8)


#plot_column(train)
#train.hist()
#data_statistics.plot_histograms(train)

#data_statistics.boxplot_rms(train, name='Training Set')
#data_statistics.boxplot_rms(test, name='Testing Set')