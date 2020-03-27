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
def plot_fft(time, vibration_signal, rot_data=[], avg_power=-1, avg_rpm=-1, interval_num=-1, name='', x_max=None):
    fast = ff_transform.FastFourierTransform(vibration_signal, time, 'gearbox')
    _, _, _, _, _, _ = fast.fft_transform_time(rot_data=rot_data, avg_power=avg_power, avg_rpm=avg_rpm, name=name, 
                                               interval_num=interval_num, plot=True, get_rms_for_bins=False, 
                                               plot_bin_lines=False, x_lim=x_max)
    return

def plot_signal(time, vibration_signal, peak_array=[], x_min=0, x_max=None, name=''):
# ------ Plot original signal -------
    '''
        x_original = []
        y_original = []
        for i in range(round_plots):
            x_original = np.append(x_original, x_interval[i])
            y_original = np.append(y_original, y_interval[i])
        original_vertical_lines = peak_array[0:round_plots+1]
    '''
    plt.figure(figsize=(15, 5))
    plt.plot(time, vibration_signal, linewidth=0.2)
    plt.title(f'{name} Vibration Data')
    plt.xlabel('Time (in s)', fontsize=16)
    plt.ylabel('Vibration amplitude (in m/s2)', fontsize=16)
    if len(peak_array) > 0:
        for i, round_value in enumerate(peak_array):
            plt.axvline(x=round_value, c='r', linewidth=0.8)
    plt.xlim(x_min, x_max)
    plt.margins(0)
    plt.show()


def plot_kurtogram(kurtogram, frequencies, file_name='', wt='', max_sk=None, cf=None, bw=None, window=None, save_path=None):
    frequencies = np.asarray(frequencies)
    kurtogram_index = ['0 (2)', '1 (4)', '1.6 (6)', '2 (8)', '2.6 (12)', '3 (16)', '3.6 (24)', '4 (32)', 
                        '4.6 (48)', '5 (64)', '5.6 (96)', '6 (128)', '6.6 (192)', '7 (256)', '7.6 (384)', '8 (512)']
    plt.figure(figsize=(10,8))
    color=['Blues_r', 'YlGnBu_r', 'YlOrRd_r', 'BuGn_r', 'GnBu_r', 'Greys_r','navy','winter','PuBu_r','CMRmap','mako', 'seismic', 'viridis']
    
    custom_blues_1= sns.light_palette((210, 90, 60), 10, input='husl', reverse=True)
    color.append(custom_blues_1)

    xticks = (frequencies)
    keptticks = xticks[::int(len(xticks)/5)]
    keptticks2 = np.floor(keptticks/1000)
    xticks = ['' for y in xticks]
    xticks[::int(len(xticks)/5)] = keptticks2

    chart = sns.heatmap(kurtogram, cmap=color[-2],yticklabels=kurtogram_index,xticklabels=xticks,cbar_kws={'label': 'Spectral Kurtosis'})
    chart.set_yticklabels(kurtogram_index, rotation=0)
    
    chart_title = f'Kurtogram'
    if file_name != '':
        chart_title = f'{chart_title} of {file_name}'
    if wt != '':
        chart_title = f'{chart_title} Turbine {wt}'
    if (max_sk is not None) and (cf is not None) and (bw is not None) and (window is not None):
        chart_title = f'{chart_title}\nMax SK value: {max_sk:.2f}     Optimal Window Length: {kurtogram_index[window]}\nCenter Frequency: {cf:.2f}         Bandwidth: {bw:.2f}'
    
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Level (Window Length)')
    plt.title(chart_title)
    if (save_path is not None) and (file_name != ''):
        plt.savefig(f'{save_path}kurt_{file_name}.png')
    #plt.show()
    plt.close()
    return None