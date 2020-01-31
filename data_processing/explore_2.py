
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data
import ff_transform
import resample
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
    cols = ['Speed Sensor;1;V', 'GnNDe;0,0102;m/s2']
    avg_speed, peak_array = get_speed_and_peaks(interval, 'Speed Sensor;1;V')
    avg_power = interval.op_df["PwrAvg;kW"][0]
    #print(f'Average Rotational Speed for {i}: {avg_speed}')
    #print(f'Average Power Generated for {i}: {avg_power}')

    time_stamps = interval.sensor_df['TimeStamp']
    vibration_signal = interval.sensor_df['GnNDe;0,0102;m/s2']
    try:
        vibration_signal = interval.sensor_df['GnNDe;0,0102;m/s2']
        time_resampled, y_resampled, all_time_resampled, all_y_resampled = resample.linear_interpolation_resampling(time_stamps,
                                                                                                  vibration_signal,
                                                                                                  peak_array, 1500,
                                                                                                  round_plots=2,
                                                                                                  plotting=False)
        #plot_sensor_data(interval, cols, avg_speed, peak_array, title=f'{i}')
    except:
        print("Could not find GnNDe;0,0102;m/s2")
        continue


    # RUN FFT on resampled data for one round
    '''
    print("FFT")
    fast = ff_transform.FastFourierTransform(y_resampled[0],time_resampled[0])
    #fast.plot_input()
    fft, time, spectral_centroid = fast.fft_transform()
    spectral_centroids.append(spectral_centroid)
    print(spectral_centroid)
    '''

    # RUN FFT on resampled data for all rounds
    print("FFT")
    fast = ff_transform.FastFourierTransform(all_y_resampled, all_time_resampled)
    # fast.plot_input()
    fft, time, spectral_centroid = fast.fft_transform(avg_speed, avg_power, i)
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

