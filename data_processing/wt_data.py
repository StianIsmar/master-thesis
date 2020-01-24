'''
This class contains all of the 10 second intervals related to a specific wind turbine. In this case, this is one of the
four wind turbines at Skomakerfjellet.
'''

import os
from process_data import TenSecondInterval
import pandas as pd
import pickle
import os.path


class Wt_data():
    # Takes on of the following as wt_name: WTG01, WTG02, WTG03, WTG04.
    def __init__(self, wt_name):
        self.name = wt_name
        self.ten_second_intervals = []
        # self.loop_directory(wt_name)
        # self.save_instance()

    def loop_directory(self, wt_name):

        path = '/Volumes/OsvikExtra/VibrationData/'
        # Count how many files in directory
        import os
        count_path = '/Volumes/OsvikExtra/VibrationData/' + wt_name
        number_of_files = 0
        for file in os.listdir(count_path):
            if not file.startswith("."):
                number_of_files+=1
        print("Number of files: ", number_of_files)

        loop_count = 0
        for filename in os.listdir(path + wt_name):
            if loop_count > 10:
                break
            if filename.endswith(".uff") and not filename[0] == ".":
                print("Files read: ", loop_count, "/", number_of_files)
                loop_count+=1
                print("Filename: " + filename)
                # print(os.path.join(directory, filename))
                interval_object = TenSecondInterval()
                interval_object.load_data(path + wt_name + "/" + filename)
                interval_object.insert_speed()

                # Added object to list of objects for particular wind turbine
                self.add_interval(interval_object)
            else:
                continue

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

    def save_instance(self):
        content = self
        pickle.dump(content, open('saved_instance_' + self.name + '.p', 'wb'))


def load_instance(name):
    wt_01 = Wt_data(name)
    wt_01 = pickle.load(open('saved_instance_' + name + '.p', 'rb'))
    return wt_01

def create_wt_data(name):
    if os.path.isfile('saved_instance_'+ name +'.p'):
        print("Pickle already exist!")
    else:
        wt_01 = Wt_data(name)
        wt_01.loop_directory(name)
        wt_01.save_instance()
        return wt_01


    # wt_02 = Wt_data()

    # wt_03 = Wt_data()

    # wt_04 = Wt_data()



