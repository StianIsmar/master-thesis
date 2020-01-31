
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data
import ff_transform
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000



''' 
Returns the avg rotational speed and an array of timestamps for each time it rotates a whole round
by calling calc_speed for an interval (from process_data.py) 
'''
def get_speed_and_peaks(interval, speed_col_name):
    avg_speed, peak_array = interval.calc_speed(speed_col_name)
    return avg_speed, peak_array

'''
Resamples vibration data by using interpolation
Input time_stamps = the x_values, vibration_signal = accelerometer measurements (y_values)
peak_array = time stamp of a shaft revolution, number_of_resample_points = desired number of resampled ponts
'''

def resample_signal_interp(time_stamps, vibration_signal, peak_array, number_of_resample_points, round_plots=0, printing=False, plotting=False):
    # Convert Panda series into numpy arrays for easier data processing
    time_stamps = np.array(time_stamps)
    vibration_signal = np.array(vibration_signal)
    peak_array = np.array(peak_array)

    # Get the index's for each shaft revolution from time_stamps
    peak_indexes = []
    for i, peak in enumerate(peak_array):
        peak_indexes.append(np.where(time_stamps == peak))
    peak_indexes = np.array(peak_indexes).flatten()

    # Extract the every revolution interval, both from time_stamps and vibration_signal
    x_interval = []
    y_interval = []
    for i in range(len(peak_indexes)-1):
        one_x_interval = time_stamps[peak_indexes[i]:peak_indexes[i+1]-1]
        one_y_interval = vibration_signal[peak_indexes[i]:peak_indexes[i + 1] - 1]
        x_interval.append(one_x_interval)
        y_interval.append(one_y_interval)

    # Convert to numpy arrays for easier data processing
    x_interval = np.array(x_interval)
    y_interval = np.array(y_interval)

    resampled_x_values = []
    resampled_y_values = []
    X_values_round_domain_list = []
    for i, one_interval in enumerate(x_interval):
        # Resample the x-coordinates from the beginning of a revolution to its end with the specified number of data points
        start_peak = x_interval[i][0]
        end_peak   = x_interval[i][-1]
        resampled_x = np.linspace(start_peak, end_peak, number_of_resample_points)

        # Resample the vibration data by linear interpolation
        resampled_y = np.interp(resampled_x, x_interval[i], y_interval[i])

        # Transform the x values from time domain to radians domain
        delta_x = resampled_x[1] - resampled_x[0]
        round_start = 2*np.pi * i
        round_end = 2*np.pi * i + 2*np.pi - delta_x
        X_values_round_domain = np.linspace(round_start, round_end, number_of_resample_points)

        # Collect the transformed lists
        resampled_x_values.append(resampled_x)
        resampled_y_values.append(resampled_y)
        X_values_round_domain_list.append(X_values_round_domain)

    resampled_x_values = np.array(resampled_x_values)
    resampled_y_values = np.array(resampled_y_values)
    X_values_round_domain_list = np.array(X_values_round_domain_list)

    # Print various values to check if everything is as it should
    if printing:
        print(f'First original x_interval value: {x_interval[0][0]}')
        print(f'Last  original x_interval value: {x_interval[0][-1]}')
        print(f'Whole original x_interval: {x_interval[0]}')
        print(f'Shape of original x_interval: {x_interval[0].shape}')

        print(f'\nFirst resampled x_interval value: {resampled_x_values[0][0]}')
        print(f'Last  resampled x_interval value: {resampled_x_values[0][-1]}')
        print(f'Shape of resampled x_interval: {resampled_x_values[0].shape}')

        print(f'\nFirst original vibration value: {y_interval[0][0]}')
        print(f'Last  original vibration value: {y_interval[0][-1]}')
        print(f'Shape of first original vibration set: {y_interval[0].shape}')

        print(f'\nFirst resampled vibration value: {resampled_y_values[0][0]}')
        print(f'Last  resampled vibration value: {resampled_y_values[0][-1]}')
        print(f'Shape of first resampled vibration set: {resampled_y_values[0].shape}')

        print(f'\nFirst ronund domain x value: {X_values_round_domain_list[0][0]}')
        print(f'Last  ronund domain x value: {X_values_round_domain_list[0][-1]}')
        print(f'Shape of ronund domain x values: {X_values_round_domain_list[0].shape}')


    if plotting:
        # ------ Plot original signal -------
        x_original = []
        y_original = []
        for i in range(round_plots):
            x_original = np.append(x_original, x_interval[i])
            y_original = np.append(y_original, y_interval[i])
        original_vertical_lines = peak_array[0:round_plots+1]

        plt.figure(figsize=(20, 10))
        plt.plot(x_original, y_original, c='b', linewidth=0.1)
        plt.title(f'Original Vibration Data. Number of Data Points: {x_original.shape[0]}', fontsize=20)
        plt.xlabel('Time (in s)', fontsize=16)
        plt.ylabel('Vibration amplitude (in m/s2)', fontsize=16)
        for i, round_value in enumerate(original_vertical_lines):
            plt.axvline(x=round_value, c='r', linewidth=0.3)
        plt.margins(0)
        plt.show()


        # ------ Plot resampled signal ------
        x_resampled = []
        y_resampled = []
        resampled_vertical_lines = []
        for i in range(round_plots):
            x_resampled = np.append(x_resampled, X_values_round_domain_list[i])
            y_resampled = np.append(y_resampled, resampled_y_values[i])
            resampled_vertical_lines.append(X_values_round_domain_list[i][0])

        plt.figure(figsize=(20, 10))
        plt.plot(x_resampled, y_resampled, c='b', linewidth=0.1)
        plt.title(f'Resampled Vibration Data. Number of Data Points: {x_resampled.shape[0]}', fontsize=20)
        plt.xlabel('Rounds (in radians)', fontsize=16)
        plt.ylabel('Vibration amplitude (in m/s2', fontsize=16)
        for i, round_value in enumerate(resampled_vertical_lines):
            plt.axvline(x=round_value, c='r', linewidth=0.3)
        plt.margins(0)
        plt.show()



    return X_values_round_domain_list, resampled_y_values



def plot_sensor_data(interval, colName, avg_speed, peak_array, title=""):
    x_values  = interval.sensor_df['TimeStamp']
    dataframe = interval.sensor_df.drop(columns=['TimeStamp'])

    delta_t = x_values[1] - x_values[0]
    samples_between_peaks = []
    for i in range(len(peak_array)-1):
        samples_between_two_peaks = (peak_array[i+1] - peak_array[i]) / delta_t
        samples_between_peaks.append(samples_between_two_peaks)
    print(f'Delta t is {delta_t}')
    print(samples_between_peaks)
    print(f'Lowest amount of samples between two peaks: {min(samples_between_peaks)}')

    peak_array = np.array(peak_array) * avg_speed
    x_values = x_values * avg_speed
    x_values -= peak_array[0]
    vertical_lines = np.arange(np.ceil(max(x_values)))

    for i, col in enumerate(colName):
        plt.figure(figsize=(20, 10))
        y_values = dataframe[col]
        # plt.plot(x_values, y_values, linewidth=0.1)
        plt.plot(x_values[0:500], dataframe[col][0:500], linewidth=0.1)
        y_mean = np.mean(y_values)
        if title != "":
            plt.title(col + ' VS Round for ' + title + f'. Power Generated: {interval.op_df["PwrAvg;kW"][0]}. Mean value: {y_mean}')
        else:
            plt.title(col + ' VS Round')
        plt.ylabel(col)
        plt.xlabel('Rounds')
        for i, x_val in enumerate(vertical_lines):
            if x_val > x_values[500]:
                break
            plt.axvline(x=x_val, c='r', linewidth=0.3)
        plt.margins(0)
        plt.show()




'''
content = pickle.load(open('saved_dfs.p', 'rb'))

op_df = content['op_df']
sensor_data_df = content['sensor_data_df']

print(sensor_data_df.columns.values)

plot_data(sensor_data_df, 'Speed Sensor;1;V')

plot_data(sensor_data_df, 'LssShf;1;V')

low_rot_speed = calc_avg_speed(sensor_data_df, 'LssShf;1;V')
high_rot_seed = calc_avg_speed(sensor_data_df, 'Speed Sensor;1;V')

print('Low rotational speed is:   ', low_rot_speed)
print('High rotational speed is:  ', high_rot_seed)
print(op_df)
print(op_df.shape)

op_df.insert(len(op_df.columns.values), "LowSpeed:rps", low_rot_speed)
op_df.insert(len(op_df.columns.values), "HighSpeed:rps", high_rot_seed)

print(op_df)
print(op_df.shape)
'''



#instance = process_data.TenSecondInterval()

#instance_path = '/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff'
#instance.load_data(instance_path)
#instance.insert_speed()
#instance.save_instance()


# ------- TO LOAD ONE DATA INSTANCE ---------
#instance = process_data.load_instance()




# --------- TO CREATE WT INSTANCES --------------

#wt_instance_1 = wt_data.create_wt_data("WTG01")
#wt_instance_2 = wt_data.create_wt_data("WTG02")
#wt_instance_3 = wt_data.create_wt_data("WTG03")
#wt_instance_4 = wt_data.create_wt_data("WTG04")
#wt_instance_short = wt_data.create_wt_data("WTG01")


# ---------  TO LOAD WT INSTANCES --------------

#wt_instance_1 = wt_data.load_instance("WTG01")
#wt_instance_2 = wt_data.load_instance("WTG02")
#wt_instance_3 = wt_data.load_instance("WTG03")
#wt_instance_4 = wt_data.load_instance("WTG04")
wt_instance = wt_data.load_instance("WTG01",load_minimal=True)
intervals = wt_instance.ten_second_intervals

# ------- Plot high rot speed ------------
spectral_centroids = []
for i, interval in enumerate(intervals):
    if i > 0:
        break
    # print(f'\nAverage Rotational Shaft Speed for {i}: {interval.op_df["HighSpeed:rps"][0]}')
    #print(f'Average Power Generated for {i}: {interval.op_df["PwrAvg;kW"][0]}')
    cols = ['Speed Sensor;1;V', 'GnNDe;0,0102;m/s2']
    avg_speed, peak_array = get_speed_and_peaks(interval, 'Speed Sensor;1;V')
    #print(f'Average Rotational Speed for {i}: {avg_speed}')
    time_stamps = interval.sensor_df['TimeStamp']
    try:
        vibration_signal = interval.sensor_df['GnNDe;0,0102;m/s2']
        time_resampled, y_resampled = resample_signal_interp(time_stamps, vibration_signal, peak_array, 1500, round_plots=2, plotting=True)
        #plot_sensor_data(interval, cols, avg_speed, peak_array, title=f'{i}')
    except:
        print("Could not find GnNDe;0,0102;m/s2")
        continue


    # RUN FFT on resampled data
    print("FFT")
    fast = ff_transform.FastFourierTransform(y_resampled[0],time_resampled[0])
    fast.plot_input()
    fft, time, spectral_centroid = fast.fft_transform()
    spectral_centroids.append(spectral_centroid)
    print(spectral_centroid)
'''    
print("plotting spectral_centroids: ")
x = (np.arange(0,len(spectral_centroids)).tolist())
y = (spectral_centroids)

plt.ylabel("Centroid value")
plt.xlabel('Interval number')
plt.plot(x,y)
plt.show()
'''


# ------- Plot low rot speed -------------
'''
for i, interval in enumerate(intervals):
    if i > 9:
        break
    print(f'\nAverage Rotational Shaft Speed for {i}: {interval.op_df["LowSpeed:rps"][0]}')
    print(f'Average Power Generated for {i}: {interval.op_df["PwrAvg;kW"][0]}')
    cols = ['LssShf;1;V', 'MnBrg;0,0102;m/s2']#, 'GbxRotBrg;0,0102;m/s2']
    plot_sensor_data(interval, cols, title=f'{i}')
'''

