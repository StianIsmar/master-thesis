import numpy as np
import sys, os

ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/data_processing/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)

import resample

'''
INPUT: 
time:              array of time stamps from 0 to end of signal
vibration_signal:  array with amplitude of vibration signal
peak_array:        array consisting of time stamps for each shaft revolution
number_of_resample_points:  integer, fixed size of number of data points in between each shaft revolution. Default is 2000
interpolation:     string: choose between linear or cubic interpolation method. Cubic is more acurate, but computationally heavy. Defaul is linear
OUTPUT:
x_round:           array of resampled x-axis in terms of rounds
resampled_y:       array of resampled amplitude
x_time:            array of resampled x-axis in terms of time
'''

def do_linear_resampling(time, vibration_signal, peak_array, number_of_resample_points=2000):
    _, _, x_round, resampled_y, x_time = resample.linear_interpolation_resampling(time,
                                                                                  vibration_signal,
                                                                                  peak_array, 
                                                                                  number_of_resample_points=number_of_resample_points)

    return x_round, resampled_y, x_time

def do_cubic_resampling(time, vibration_signal, peak_array, number_of_resample_points=2000):
    _, _, x_round, resampled_y, x_time = resample.cubic_interpolation_resampling(time,
                                                                                 vibration_signal,
                                                                                 peak_array, 
                                                                                 number_of_resample_points=number_of_resample_points)

    return x_round, resampled_y, x_time