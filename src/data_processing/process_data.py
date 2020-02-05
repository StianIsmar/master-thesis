'''
This module contains the class TenSecondInterval, which reads and transforms a uff file to a pandas dataframe.
This class is called in module wt_data.py when reading all data.
'''


import pyuff
import pprint
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import gc

class TenSecondInterval:
    def __init__(self):
        self.sensor_df = None # Dataframe for sensor data
        self.op_df = None # Dataframe for operational data
        self.date = None
        self.turbine = None
        self.low_speed_rot_data = None
        self.high_speed_rot_data = None
        self.low_speed_peak_array = None
        self.high_speed_peak_array = None


    def load_data(self, path):
        pp = pprint.PrettyPrinter(indent=4)


        uff_file = pyuff.UFF(path)
        data = uff_file.read_sets()

        date    = data[0]['id3']  # Extract the date and time for operation
        turbine = data[0]['id4']  # Extract info of which turbine data belongs to

        timeStamp         = []  # Placeholder for time-stamp
        sensor_name       = []  # Stores sensor name
        sensor_data       = []  # Stores sensor data
        op_condition_name = [] # Stores name for operational condition
        op_condition_data = [] # Stores operational condition data, i.e. Average Power for the 10 second interval

        for i in range(len(data)): # Looping through each "variable" in i=0 to 178
            name = data[i]['id5']
            info = (data[i]['data'])

            if len(info) > 1:   # only add sensor data which has some measurements
                sensor_name.append(name)
                sensor_data.append(info)

                if len(timeStamp) == 0:   # timestamp is the same for all sensor data, so load it only once
                    timeStamp.append(data[i]['x'])
                    timeStamp = timeStamp[0]

            else: # Append to the operational information
                op_condition_name.append(name)
                op_condition_data.append(info)

        # Create a dataframe for the sensor data
        sensor_data_np = np.array(sensor_data)  # create a numpy array to be able to transpose it and get it in the right order
        sensor_data_df = pd.DataFrame(sensor_data_np.T, columns=sensor_name)
        sensor_data_df.insert(0, column='TimeStamp', value=timeStamp)

        # Create a dataframe for the operational condition data
        op_data_np = np.array(op_condition_data)
        op_data_df = pd.DataFrame(op_data_np.T, columns=op_condition_name)


        #pp.pprint(data[2]) # used to print a raw sensor measurement

        del sensor_data_np, op_data_np, data, info, sensor_data

        gc.collect()

        # Setting the class variables
        self.sensor_df = sensor_data_df
        self.date = date
        self.turbine = turbine
        self.op_df = op_data_df


    def plot_data(self, dataframe):
        x_values = dataframe['TimeStamp']
        dataframe = dataframe.drop(columns=['TimeStamp'])
        sensor_name = list(dataframe.columns.values)

        for i in range(len(sensor_name)):
            if sensor_name[i] == "LssShf;1;V":  # Plot the whole 10 sec for this sensor.
                plt.plot(x_values, dataframe[sensor_name[i]], linewidth=0.1)
            else:
                plt.plot(x_values[0:12500], dataframe[sensor_name[i]][0:12500], linewidth=0.1)
            plt.title(sensor_name[i] + ' VS time')
            plt.ylabel(sensor_name[i])
            plt.xlabel('Time')
            plt.show()

    def calc_speed(self, col_name):
        # Use try/except to deal with not finding the right column
        try:
            x = np.array(self.sensor_df['TimeStamp'])
            data_list = np.array(self.sensor_df[col_name])
            first_pulse_time = None  # Remember when the first pulse starts
            rotations = []  # Store the time it takes to do one revolution (this is seconds per round)
            peaks = []   # Store the time for each peak
            if data_list[0] > 20:  # If the timeserie start at the peak we record its corresponding time
                first_pulse_time = x[0]
                peaks.append(first_pulse_time)

            for i in range(len(data_list)):
                if (data_list[i] > 20) and (data_list[i - 1] < 20):
                    if first_pulse_time is None:  # Find start time of the first pulse
                        first_pulse_time = x[i]
                        peaks.append(first_pulse_time)
                    else:
                        second_pulse_time = x[i]
                        rotations.append(second_pulse_time - first_pulse_time)  # Calculate the time to do one revolution and append it to the list
                        first_pulse_time = second_pulse_time
                        peaks.append(first_pulse_time)

            # Convert the list of rotations into rounds per minute

            if rotations[0] == 0:   # Check if the fist element is zero to avoid division by zero
                rps = 1 / np.array(rotations[1:])   # Rounds per second
            else :
                rps = 1 / np.array(rotations)

            rpm = 60 * rps                  # Rounds per minute

            # If the data measurements does not contain a whole rotation, set the speed to None
            if len(rpm) > 0:
                rot_data = {
                    'mean' : np.mean(rpm),
                    'std'  : np.std(rpm),
                    'min'  : np.min(rpm),
                    'max'  : np.max(rpm)
                }
            else:
                rot_data = None

        except Exception:
            rot_data = None
            peaks = None

        return rot_data, peaks

    '''
    Calculates the speed data (mean, max, min and std) for the slow rotating shaft and the fast rotating shaft
    Extracts the peak array (time for each rotational pulse)
    '''
    def insert_speed(self):
        low_rot_data, low_peak_array = self.calc_speed('LssShf;1;V')
        high_rot_data, high_peak_array = self.calc_speed('Speed Sensor;1;V')
        self.low_speed_rot_data = low_rot_data
        self.low_speed_peak_array = low_peak_array
        self.high_speed_rot_data = high_rot_data
        self.high_speed_peak_array = high_peak_array


    def save_instance(self):
        content = self
        path = '/Volumes/OsvikExtra/VibrationData/'
        pickle.dump(content, open(path + 'saved_instance.p', 'wb'))


def load_instance():
    content = TenSecondInterval()
    path = '/Volumes/OsvikExtra/VibrationData/'
    content = pickle.load(open(path + 'saved_instance.p', 'rb'))
    return content

# ------ Example for WT01 -------
#interval = TenSecondInterval()
#interval.load_data('/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')
#print(interval.date) # Printing date

#interval.insert_speed()
#print(interval.op_df)
#print(interval.low_speed_peak_array)
#interval.plot_data(interval.sensor_df)
#print(interval.sensor_df)
#interval.save_df()
#interval.load_df()
# data, name, date, turbine = load_data('/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')

#interval.save_instance()
#instance = load_instance()
