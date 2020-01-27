'''
import process_data

interval = process_data.interval


print(interval.date)

'''
import pickle

import dateutil.parser
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.dates as dates
import datetime
from pandas.plotting import register_matplotlib_converters

import time


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
    first_pulse_time = None  # Remember when the first pulse starts
    rotations = []  # Store the time it takes to do one revolution
    if data_list[0] > 20:  # If the timeserie start at the peak we record its corresponding time
        first_pulse_time = x[0]

    for i in range(len(data_list)):
        if (data_list[i] > 20) and (data_list[i - 1] < 20):
            if first_pulse_time is None:  # Find start time of the first pulse
                first_pulse_time = x[i]
            else:
                second_pulse_time = x[i]
                # Calculate the time to do one revolution and append it to the list
                rotations.append(second_pulse_time - first_pulse_time)
                first_pulse_time = second_pulse_time

    avg_rotation = sum(rotations) / len(rotations)  # This is seconds per rotation
    avg_rotation_per_sec = 1 / avg_rotation  # This gives rotation per second
    return avg_rotation_per_sec


# Plotting all intervals for a certain wt:
def build_op_df_for_wt(wt_obj):
    wt_name = wt_obj.name  # Name of the windturbine
    variables = wt_obj.ten_second_intervals[0].op_df.columns  # All op variables for the ten_second_intervals

    # Defining the dataframe to hold all interval op data
    df_op_wt = pd.DataFrame(columns=variables)
    dates = []

    # Loop through all 10-second intervals from wind turbine
    for i, interval in enumerate(wt_obj.ten_second_intervals):
        interval.date = "20" +interval.date
        date = dateutil.parser.isoparse(interval.date)

        # Create datetime object
        datetime_obj = datetime.datetime(date.year, date.month,date.day,date.hour,date.minute,date.second)
        dates.append(datetime_obj)

        # Inserting the row for each interval
        # name = interval.name
        row = interval.op_df.iloc[0, :]
        df_op_wt = df_op_wt.append(row.T)

    # All rows are now added to dataframe. Now, add date list
    df_op_wt.insert(0, column='Date', value=dates)  # Insert the first column to hold date (for the 10 second interval)
    return df_op_wt


def plot_op_df(op_dataframe):
    # Plot in this loop


    date_series = op_dataframe['Date']

    # Looping through the columns in the dataframe
    for i in range(1, op_dataframe.shape[1]):
        if (i > 40):
            break;
        variable_name = op_dataframe.columns[i]
        register_matplotlib_converters() # From import to convert from timestamp objects to datateim
        x_dates = pd.to_datetime(date_series)

       # x_dates = dates.date2num(x_dates)
        # x_dates = date_series

        # y-axis
        y = op_dataframe.loc[:, variable_name]

        sns.lineplot(x_dates, y)
        plt.xticks(rotation='vertical')

        plt.show()

# wt_instance_1 = wt_data.load_instance("WTG01", True) # True for loading minimal
wt_instance_1 = wt_data.load_instance("WTG01", False) # True for loading minimal

print("Building op dataframe...")
df = build_op_df_for_wt(wt_instance_1)
print("Plotting dataframe...")
plot_op_df(df)


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

# instance = process_data.TenSecondInterval()

# instance = process_data.load_instance()
# print(instance.date)



# --------- TO CREATE WT INSTANCES --------------

# wt_instance_1 = wt_data.create_wt_data("WTG01")
# wt_instance_2 = wt_data.create_wt_data("WTG02")
# wt_instance_3 = wt_data.create_wt_data("WTG03")
# wt_instance_4 = wt_data.create_wt_data("WTG04")


# ---------  TO LOAD WT INSTANCES --------------

# wt_instance_1 = wt_data.load_instance("WTG01")
# wt_instance_2 = wt_data.load_instance("WTG02")
# wt_instance_3 = wt_data.load_instance("WTG03")
# wt_instance_4 = wt_data.load_instance("WTG04")



