'''
This class contains all of the 10 second intervals related to a specific wind turbine. In this case, this is one of the
four wind turbines at Skomakerfjellet.
'''

import os
from ten_second_interval import TenSecondInterval
import pandas as pd
import pickle
import os.path
import os


class Wt_data:
    # Takes on of the following as wt_name: WTG01, WTG02, WTG03, WTG04.
    def __init__(self, wt_name):
        self.name = wt_name
        self.ten_second_intervals = []
        # self.loop_directory(wt_name)
        # self.save_instance()

# Only loading 10 of the intervals with save_minimal set to True, for development purposes
    def loop_directory(self, wt_name, save_minimal = False):
        path = '/Volumes/OsvikExtra/VibrationData/'
        # Count how many files in directory

        count_path = '/Volumes/OsvikExtra/VibrationData/' + wt_name
        number_of_files = 0
        for file in os.listdir(count_path):
            if not file.startswith("."):
                number_of_files+=1
        print(f"\nNumber of files for {self.name}: {number_of_files}")

        loop_count = 0
        for filename in os.listdir(path + wt_name):
            print(loop_count)
            if save_minimal == True:
                if loop_count > 10:
                    break
            if filename.endswith(".uff") and not filename[0] == ".":
                loop_count+=1
                #print("Filename: " + filename)
                # print(os.path.join(directory, filename))
                interval_object = TenSecondInterval()
                interval_object.load_data(path + wt_name + "/" + filename)
                interval_object.insert_speed()
                print(loop_count)

                # Added object to list of objects for particular wind turbine
                self.add_interval(interval_object)
                print(f"Files read: {loop_count} / {number_of_files}", end="\r")
            else:
                continue
        print(f"Files read: {loop_count-1} / {number_of_files}")
        print(f"Completed reading of {self.name}")

    def add_interval(self, interval):
        self.ten_second_intervals.append(interval)

    # Method for combining the operational data
    def combine_op_data(self):
        for interval, i in enumerate(self.ten_second_intervals):
            print(i)
            op_df = interval.op_df
            if i == 0:
                wt_op_df = pd.DataFrame(op_df.iloc[0,:], columns = op_df.columns)
            else:
                wt_op_df.append(op_df.iloc[0,:])
                print(wt_op_df.shape)

    def save_instance(self,save_minimal=False):
        content = self
        path = '/Volumes/OsvikExtra/VibrationData/'

        if (save_minimal == True):
            pickle.dump(content, open(path + 'saved_instance_MINIMAL_' + self.name + '.p', 'wb'))
            print(f'Saved **MINIMAL INTEVALS** to file')
        else:

            pickle.dump(content, open(path + 'saved_instance_' + self.name + '.p', 'wb'))
            print(f'Saved to file')


def load_instance(name, load_minimal=False):
    print(f'\nLoading {name}...', end='\r')
    wt_01 = Wt_data(name)
    path = '/Volumes/OsvikExtra/VibrationData/'
    if (load_minimal == True):
        wt_01 = pickle.load(open(path + 'saved_instance_MINIMAL_' + name + '.p', 'rb'))
    else:
        wt_01 = pickle.load(open(path + 'saved_instance_' + name + '.p', 'rb'))
        print(f'\nLoaded {name}')
    return wt_01

def create_wt_data(name, save_minimal=False):
    if (save_minimal == True):
        wt_01 = Wt_data(name)
        wt_01.loop_directory(name, True)
        wt_01.save_instance(True)
    else:
        wt_01 = Wt_data(name)
        wt_01.loop_directory(name)
        wt_01.save_instance()
    return wt_01

#create_wt_data('WTG01', save_minimal=True)

# wt_instance_1 = create_wt_data("WTG01", True) # Only loading 10 of the intervals with True argument

'''
wt_instance_1 = create_wt_data("WTG01", True) # Only loading 10 of the intervals with True argument
saved = False
for i, instance in enumerate(wt_instance_1.ten_second_intervals):

    if (not (saved == True)):
        print("saved")
        df = instance.sensor_df
        df.to_csv(r'instance.sensor_df.csv')
        saved = True
'''


'''
WIND_TURBINE = 'WTG01'
SENSOR_NAME = 'GbxHssRr;0,0102;m/s2'
wt_instance = load_instance(WIND_TURBINE, load_minimal=True)
interval = wt_instance.ten_second_intervals[0]

rot_data = interval.high_speed_rot_data
avg_power = interval.op_df["PwrAvg;kW"][0]
BINS = 50
lower_range_freq = 0
higher_range_freq = 2200
comp_type = 'gearbox'
ts = interval.sensor_df['TimeStamp']  # Have this as the y-axis to see how the RMS/frequencies develop


vibration_signal = interval.sensor_df[SENSOR_NAME]

fast = ff_transform.FastFourierTransform(vibration_signal, ts, comp_type)
fft, time, centroid, rms, rms_bins, bin_freq = fast.fft_transform_time(
    rot_data,
    avg_power,
    get_rms_for_bins=True,
    plot=False,
    bins=BINS,
    plot_bin_lines=False,
    x_lim=False,
    frequency_lines=[],
    horisontal_lines=[],
    spectrum_lower_range=lower_range_freq,
    spectrum_higher_range=higher_range_freq
)
'''



# wt_02 = Wt_data()

# wt_03 = Wt_data()

# wt_04 = Wt_data()



