'''
import process_data

interval = process_data.interval


print(interval.date)

'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data

def plot_sensor_data(interval, colName, title=""):
    x_values   = interval.sensor_df['TimeStamp']
    dataframe  = interval.sensor_df.drop(columns=['TimeStamp'])
    low_speed  = interval.op_df['LowSpeed:rps'][0]
    high_speed = interval.op_df['HighSpeed:rps'][0]

    avg_low_speed, low_peak_array = interval.calc_speed('LssShf;1;V')
    avg_high_speed, high_peak_array = interval.calc_speed('Speed Sensor;1;V')

    low_peak_array = np.array(low_peak_array) * low_speed

    x_values = x_values * low_speed
    x_values -= low_peak_array[0]
    vertical_lines = np.arange(np.ceil(max(x_values)))

    for i, col in enumerate(colName):
        plt.figure(figsize=(20, 10))
        y_values = dataframe[col]
        plt.plot(x_values, y_values, linewidth=0.1)
        y_mean = np.mean(y_values)
        # plt.plot(x_values[0:15000], dataframe[col][0:15000], linewidth=0.1)
        if title != "":
            plt.title(col + ' VS Round for ' + title + f'. Power Generated: {interval.op_df["PwrAvg;kW"][0]}. Mean value: {y_mean}')
        else:
            plt.title(col + ' VS Round')
        plt.ylabel(col)
        plt.xlabel('Rounds')
        for i, x_val in enumerate(vertical_lines):
            plt.axvline(x=x_val, c='r', linewidth=0.5)
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
wt_instance = wt_data.load_instance("WTG01")
intervals = wt_instance.ten_second_intervals

# ------- Plot high rot speed ------------
#plot_data(instance.sensor_df, 'GnNDe;0,0102;m/s2', speed_rps=instance.op_df['HighSpeed:rps'][0])

# ------- Plot low rot speed -------------

for i, interval in enumerate(intervals):
    if i > 9:
        break
    print(f'\nAverage Rotational Shaft Speed for {i}: {interval.op_df["LowSpeed:rps"][0]}')
    print(f'Average Power Generated for {i}: {interval.op_df["PwrAvg;kW"][0]}')
    cols = ['LssShf;1;V', 'MnBrg;0,0102;m/s2']#, 'GbxRotBrg;0,0102;m/s2']
    plot_sensor_data(interval, cols, title=f'{i}')
    #plot_data(interval.sensor_df, 'LssShf;1;V', speed_rps=interval.op_df['LowSpeed:rps'][0], title=f'{i}')
    #plot_data(interval.sensor_df, 'MnBrg;0,0102;m/s2', speed_rps=interval.op_df['LowSpeed:rps'][0], title=f'{i}')
    #plot_data(interval.sensor_df, 'GbxRotBrg;0,0102;m/s2', speed_rps=interval.op_df['LowSpeed:rps'][0], title=f'{i}')


