
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data
import ff_transform
import resample
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000



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

#wt_instance = wt_data.load_instance("WTG01")
#wt_instance_2 = wt_data.load_instance("WTG02")
#wt_instance_3 = wt_data.load_instance("WTG03")
#wt_instance_4 = wt_data.load_instance("WTG04")
#wt_instance_1 = wt_data.create_wt_data("WTG01", True)
wt_instance = wt_data.load_instance("WTG01",load_minimal=True)
intervals = wt_instance.ten_second_intervals

# ------- Plot high rot speed ------------
spectral_centroids = []
for i, interval in enumerate(intervals):
    if i > 20:
        break
    print(f'Checking interval: {i}', end='\r')
    cols = ['Speed Sensor;1;V', 'GnNDe;0,0102;m/s2']
    rot_data = interval.high_speed_rot_data
    peak_array = interval.high_speed_peak_array
    avg_speed = rot_data['mean']
    avg_power = interval.op_df["PwrAvg;kW"][0]

    if avg_power > 2500:
        print(f'Calc FFT for interval: {i}', end='\r')
        time_stamps = interval.sensor_df['TimeStamp']
        try:
            vibration_signal = interval.sensor_df['GbxHssRr;0,0102;m/s2']
            time_resampled, y_resampled, all_time_resampled, all_y_resampled = resample.linear_interpolation_resampling(time_stamps,
                                                                                                      vibration_signal,
                                                                                                      peak_array, 1500,
                                                                                                      round_plots=2,
                                                                                                      plotting=False)
            #plot_sensor_data(interval, cols, avg_speed, peak_array, title=f'{i}')
        except:
            print("Could not find GnNDe;0,0102;m/s2")
            continue


        # RUN FFT on resampled data for one revolution
        '''
        print("FFT")
        fast = ff_transform.FastFourierTransform(y_resampled[0],time_resampled[0])
        #fast.plot_input()
        fft, time, spectral_centroid = fast.fft_transform()
        spectral_centroids.append(spectral_centroid)
        print(spectral_centroid)
        '''

        # RUN FFT on resampled data for all revolutions
        fast = ff_transform.FastFourierTransform(all_y_resampled, all_time_resampled)
        # fast.plot_input()
        fft, time, spectral_centroid = fast.fft_transform_order(rot_data, avg_power, i, plot=True)
        spectral_centroids.append(spectral_centroid)
        #print(spectral_centroid)

''' 
print("plotting spectral_centroids: ")
x = (np.arange(0,len(spectral_centroids)).tolist())
y = (spectral_centroids)

plt.ylabel("Centroid value")
plt.xlabel('Interval number')
plt.plot(x,y)
plt.show()
'''
