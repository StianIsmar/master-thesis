'''
This module contains the class TenSecondInterval, which reads transforms a uff file to a pandas dataframe.
Call module from wt_data.py to load all data.
'''


import pyuff
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


class TenSecondInterval:
    def __init__(self):
        self.df = None
        self.name = None
        self.date = None
        self.turbine = None

    def load_data(self, path):
        print ("Start read...\n")
        pp = pprint.PrettyPrinter(indent=4)

        try:
            uff_file = pyuff.UFF(path)
            data = uff_file.read_sets()

            date    = data[0]['id3']  # Extract the date and time for operation
            turbine = data[0]['id4']  # Extract info of which turbine data belongs to

            timeStamp     = []  # Placeholder for time-stamp
            sensor_name   = []  # Stores sensor name
            sensor_data   = []  # Stores sensor data

            for i in range(len(data)):
                name = data[i]['id5']
                info = (data[i]['data'])

                if len(info) > 1:   # only add sensor data which has some measurements
                    sensor_name.append(name)
                    sensor_data.append(info)

                    if len(timeStamp) == 0:   # timestamp is the same for all sensor data, so load it only once
                        timeStamp.append(data[i]['x'])
                        timeStamp = timeStamp[0]

            data_np = np.array(sensor_data)  # create a numpy array to be able to transpose it and get it in the right order
            data_pd = pd.DataFrame(data_np.T, columns=sensor_name)
            data_pd.insert(0, column='TimeStamp', value=timeStamp)


            print(f"Loaded turbine {turbine} for date {date}.")
            print('----------------------------------------------------------------------\n')
            #pp.pprint(data[2]) # used to print a raw sensor measurement

            # Setting the class variables
            self.df = data_pd
            self.name = name
            self.date = date
            self.turbine = turbine

        except Exception as e:
            logging.exception(e)
            print(f" Error loading 10 second interval data for path = {path}.")

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


# Example for WT01:
interval = TenSecondInterval()
# interval.load_data('/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')
interval.load_data('/Users/stian/Desktop/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')
print(interval.date) # Printing date
interval.plot_data(interval.df)
print(interval.df)
# data, name, date, turbine = load_data('/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')

