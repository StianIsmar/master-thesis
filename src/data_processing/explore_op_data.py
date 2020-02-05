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
import gc

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

    print("df shape for building dataframe: ", df_op_wt.shape)
    return df_op_wt


def plot_op_df(op_dataframe):
    # Plot in this loop

    date_series = op_dataframe['Date']

    # Looping through the columns in the dataframe
    for i in range(1, op_dataframe.shape[1]):
        print(i)
        variable_name = op_dataframe.columns[i]
        register_matplotlib_converters()  # From import to convert from timestamp objects to datateim
        x_dates = pd.to_datetime(date_series)

       # x_dates = dates.date2num(x_dates)
        # x_dates = date_series

        # y-axis
        y = op_dataframe.loc[:, variable_name]

        sns.lineplot(x_dates, y)
        plt.xticks(rotation='vertical')

        def remove_back(variable):
            check = variable.find('/')
            if check == -1:
                return variable
            else:
                variable = variable.replace('/', '_')
                remove_back(variable)

        # variable_name = remove_back(variable_name)
        variable_name = variable_name.replace('/', '_')
        plt.savefig(f'./plotting/op_plot_{variable_name}.png')
        plt.show()


def draw_correlation_plot(df):
    print("df shape for correlation plot: ", df.shape)
    sns.set(style="white")
    df = df.drop(columns = ["Date"])

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # cmap = sns.light_palette("#2ecc71", as_cmap=True)
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)


    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # saving the file
    f.tight_layout()
    f.savefig('./plotting/correlation_plot.eps', format='eps')

def get_min_max_from_column(df, column_name):
    print(df[column_name].max())
    print(df[column_name].min())


def remove_irrelevant_features(df):

    '''
    print("Before constant clean", df.shape)
    drop_these = df.columns[df.nunique() <= 1]
    df.drop(drop_these, axis=1, inplace=True)
    print("After constant clean", df.shape)
    '''

    print("Before unique clean: ", df.shape)
    n_cols_before = df.shape[1]
    for i, col in enumerate(df.columns):
        if df[col].nunique() < 6:
            df.drop(col, axis=1, inplace=True)
    print("After unique cleand: ", df.shape)
    n_cols_after = df.shape[1]
    print("Number of columns removed: ", n_cols_before-n_cols_after)
    return df


def remove_rows_with_wild_noise(df):
    print("Shape before removing noise", df.shape)
    i = 0
    for cols in (df.columns):
        if i == 0:  # Skip the first row, which is datetime objects
            i=i+1
            continue
        df = df[df[cols] > -10**15]
        i=i+1

    print("Shape after removing noise", df.shape)
    return df


def load_and_build_df(name):
    wt_instance_1 = wt_data.load_instance(name, True)  # True for loading minimal
    gc.collect()
    op_df = build_op_df_for_wt(wt_instance_1)
    del wt_instance_1
    op_df = remove_rows_with_wild_noise(op_df)
    gc.collect()
    # get_min_max_from_column(op_df, "PwrAct;kW")
    # Removing irrelevant features
    op_df = remove_irrelevant_features(op_df)
    gc.collect()
    draw_correlation_plot(op_df)
    plot_op_df(op_df)
    return op_df

# --------- LOAD WT INSTANCE AND BUILD DATAFRAME (OP DATA) --------------


op_df_WTG01 = load_and_build_df("WTG01")


