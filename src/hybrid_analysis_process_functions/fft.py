import numpy as np
import sys, os

ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/data_processing/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)

import ff_transform

'''
INPUT: 
time:              array of timestamps
vibration_signal:  array of amplitude of vibration signal
rot_data:          dictionary - contains operational data like mean rpm.
avg_power:         float - average power generated for interval
interval_num:      int - which interval is printed
name:              string - additional info to the figure header. Default is ''.
OUTPUT:
nothing, only displays figure. 
'''
def plot_fft(time, vibration_signal, rot_data=[], avg_power=-1, avg_rpm=-1, interval_num=-1, name='',):
    fast = ff_transform.FastFourierTransform(vibration_signal, time, 'gearbox')
    _, _, _, _, _, _ = fast.fft_transform_time(rot_data=rot_data, avg_power=avg_power, avg_rpm=avg_rpm, name=name, interval_num=interval_num, plot=True, get_rms_for_bins=False, plot_bin_lines=False)
    return