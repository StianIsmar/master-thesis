'''
import process_data

interval = process_data.interval


print(interval.date)

'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import process_data

def plot_data(dataframe, colName, plot_all=False):
    x_values = dataframe['TimeStamp']
    dataframe = dataframe.drop(columns=['TimeStamp'])
    sensor_name = list(dataframe.columns.values)

    if plot_all:
        for i in range(len(sensor_name)):
            if sensor_name[i] == "LssShf;1;V":  # Plot the whole 10 sec for this sensor.
                plt.plot(x_values, dataframe[sensor_name[i]], linewidth=0.1)
            else:
                plt.plot(x_values[0:12500], dataframe[sensor_name[i]][0:12500], linewidth=0.1)

            plt.title(sensor_name[i] + ' VS time')
            plt.ylabel(sensor_name[i])
            plt.xlabel('Time')
            plt.show()

    else:
        plt.plot(x_values, dataframe[colName], linewidth=0.1)
        plt.title(colName + ' VS time')
        plt.ylabel(colName)
        plt.xlabel('Time')
        plt.show()


def calc_avg_speed(dataframe, col_name):
    x = np.array(dataframe['TimeStamp'])
    data_list = np.array(dataframe[col_name])
    first_pulse_time = None   # Remember when the first pulse starts
    rotations = []    # Store the time it takes to do one revolution
    if data_list[0] > 20:    # If the timeserie start at the peak we record its corresponding time
        first_pulse_time = x[0]

    for i in range(len(data_list)):
        if (data_list[i] > 20) and (data_list[i-1] < 20):
            if first_pulse_time is None:   # Find start time of the first pulse
                first_pulse_time = x[i]
            else:
                second_pulse_time = x[i]
                rotations.append(second_pulse_time - first_pulse_time) # Calculate the time to do one revolution and append it to the list
                first_pulse_time = second_pulse_time

    avg_rotation = sum(rotations)/len(rotations) # This is seconds per rotation
    avg_rotation_per_sec = 1/avg_rotation  # This gives rotation per second
    return avg_rotation_per_sec

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

instance = process_data.load_instance()
print(instance.date)





