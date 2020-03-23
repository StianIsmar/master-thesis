import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_signal(time, vibration_signal, x_min=0, x_max=None):

# ------ Plot original signal -------
        x_original = []
        y_original = []
        for i in range(round_plots):
            x_original = np.append(x_original, x_interval[i])
            y_original = np.append(y_original, y_interval[i])
        original_vertical_lines = peak_array[0:round_plots+1]

        plt.figure(figsize=(15, 5))
        plt.plot(x_original, y_original, linewidth=0.2)
        plt.title(f'Original Vibration Data {name} \nNumber of Data Points: {x_original.shape[0]}')
        plt.xlabel('Time (in s)', fontsize=16)
        plt.ylabel('Vibration amplitude (in m/s2)', fontsize=16)
        for i, round_value in enumerate(original_vertical_lines):
            plt.axvline(x=round_value, c='r', linewidth=0.8)
        plt.margins(0)
        plt.show()


def plot_kurtogram(kurtogram, frequencies):
    frequencies = np.asarray(frequencies)
    kurtogram_index = [0, 1, 1.6, 2, 2.6, 3, 3.6, 4, 4.6, 5, 5.6, 6, 6.6, 7, 7.6, 8]
    plt.figure(figsize=(10,8))
    color=['Blues_r', 'YlGnBu_r']
    chart = sns.heatmap(kurtogram, cmap=color[1])
    chart.set_xticklabels(frequencies, minor=False, rotation=45)
    chart.set_yticklabels(kurtogram_index, rotation=0)
    plt.show()