import sys, os
import numpy as np
import seaborn as sns
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection

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
def plot_fft(time, vibration_signal, rot_data=[], avg_power=-1, avg_rpm=-1, interval_num=-1, name='', x_min=0, x_max=None, y_max=None, plot_lim=False, frequency_lines=[], save_path=None, default_title=True):
    fast = ff_transform.FastFourierTransform(vibration_signal, time, 'gearbox')
    fft, time, _, _, _, _ = fast.fft_transform_time(rot_data=rot_data, avg_power=avg_power, avg_rpm=avg_rpm, name=name, 
                                               interval_num=interval_num, plot=False, get_rms_for_bins=False, 
                                               plot_bin_lines=False, x_min=x_min, x_max=x_max, y_max=y_max, 
                                               plot_lim=plot_lim, frequency_lines=frequency_lines, save_path=save_path,
                                               default_title=default_title)
    return fft, time

def plot_signal(time, vibration_signal, peak_array=[], avg_power=-1, avg_rpm=-1, interval_num=-1, x_min=0, x_max=None, name='', linewidth=0.2, save_path=None):
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
    plt.plot(time, vibration_signal, linewidth=linewidth)
    heading = f'{name}' 
    plt.xlabel('Time (in s)', fontsize=16)
    plt.ylabel('Vibration amplitude (in m/s2)', fontsize=16)
    if len(peak_array) > 0:
        for i, round_value in enumerate(peak_array):
            plt.axvline(x=round_value, c='r', linewidth=0.8)
    if interval_num > 0:
        heading = f'{heading}     Interval: {interval_num}'
    if avg_rpm > -1 and avg_power > -1:
        heading = f'{heading}\n     Avg Power: {avg_power:.2f}     Avg RPM: {avg_rpm:.2f}'
    plt.title(heading)
    if x_max is None:
        x_max = time[-1]
    plt.xlim(x_min, x_max)
    #plt.margins(0)
    if save_path is not None:
        plt.savefig(f'{save_path}.png', dpi=300)
    plt.show()


def plot_kurtogram(kurtogram, frequencies, file_name='', wt='', max_sk=None, cf=None, bw=None, window=None, save_path=None, title=None):
    frequencies = np.asarray(frequencies)
    kurtogram_index = ['0 (2)', '1 (4)', '1.6 (6)', '2 (8)', '2.6 (12)', '3 (16)', '3.6 (24)', '4 (32)', 
                        '4.6 (48)', '5 (64)', '5.6 (96)', '6 (128)', '6.6 (192)', '7 (256)', '7.6 (384)', '8 (512)']
    plt.figure(figsize=(10,8))
    color=['Blues_r', 'YlGnBu_r', 'YlOrRd_r', 'BuGn_r', 'GnBu_r', 'Greys_r','navy','winter','PuBu_r','CMRmap','mako', 'seismic', 'viridis']
    
    custom_blues_1= sns.light_palette((210, 90, 60), 10, input='husl', reverse=True)
    color.append(custom_blues_1)

    xticks = (frequencies)
    keptticks = xticks[::int(len(xticks)/5)]
    keptticks2 = np.floor(keptticks/100) / 10
    xticks = ['' for y in xticks]
    xticks[::int(len(xticks)/5)] = keptticks2

    chart = sns.heatmap(kurtogram, cmap=color[-2],yticklabels=kurtogram_index,xticklabels=xticks,cbar_kws={'label': 'Spectral Kurtosis'})
    chart.set_yticklabels(kurtogram_index, rotation=0)
    
    if title is None:
        chart_title = f'Kurtogram'
        if file_name != '':
            chart_title = f'{chart_title} of {file_name}'
        if wt != '':
            chart_title = f'{chart_title} Turbine {wt}'
    
    else:
        chart_title = title

    if (max_sk is not None) and (cf is not None) and (bw is not None) and (window is not None):
        chart_title = f'{chart_title}\nMax SK value: {max_sk:.2f}      Optimal Window Length: {kurtogram_index[window]}\nCenter Frequency: {cf:.2f}         Bandwidth: {bw:.2f}'
    
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Level (Window Length)')
    plt.title(chart_title)
    if (save_path is not None) and (file_name != ''):
        plt.savefig(f'{save_path}.png', dpi=300)
    plt.show()
    plt.close()
    return None


def print3d_modular(x,y,z,turbine_number,sensor_name,plot_title, z_max_plot, colorbar_val_min, colorbar_val_max, x_axis_title, average_powers, cm_style='Blues'):
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{plot_title} {turbine_number}",pad=20)
    
    # Get the numpy arrays on the correct shape
    freq_data = x.T
    amp_data = z.T
    rad_data = np.linspace(0,amp_data.shape[1],amp_data.shape[1])
    verts = []

    for irad in range(len(rad_data)):
        # I'm adding a zero amplitude at the beginning and the end to get a nice
        # flat bottom on the polygons
        xs = np.concatenate([[freq_data[0,irad]], freq_data[:,irad], [freq_data[-1,irad]]])
        ys = np.concatenate([[0],amp_data[:,irad],[0]])
        verts.append(list(zip(xs, ys)))

    # Colors:
    
    # cmap="coolwarm"
    cmap=cm_style # copper or Blues works
    cmap = cmx.get_cmap(cmap)
    scaled = minmax_scale(average_powers)
    col = [cmap(x) for x in scaled]
    #col= ['r' for x in np.ones(len(average_powers))*0.85]
    print(map)
    poly = PolyCollection(verts,facecolors=col, edgecolors=col)

    
    #poly.set_alpha(0.7)

    # The zdir keyword makes it plot the "z" vertex dimension (radius)
    # along the y axis. The zs keyword sets each polygon at the
    # correct radius value.
    ax.add_collection3d(poly, zs=rad_data, zdir='y')
    ax.set_xlim3d(0, freq_data.max())
    # ticks = np.arange(0, freq_data.max(), 250)
    #ax.set_xticks(ticks)
    
    # ax.set_xticks(np.arange(0,freq_data.max(),10))
    ax.set_xlabel(f'{x_axis_title}',labelpad=10)
    ax.set_ylim3d(rad_data.min(), rad_data.max())
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)


    ax.set_ylabel('Interval number',labelpad=10)
    # ax.set_zlim3d(amp_data.min(), amp_data.max())
    ax.set_zlim3d(0, z_max_plot)
    ax.set_zlabel(f' {sensor_name} RMS amplitude')

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    mn=int(colorbar_val_min)  
    mx=int(colorbar_val_max)
    md=(mx-mn)/2
    cb = plt.colorbar(sm)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels([mn,md,mx],update_ticks=True)
    cb.set_label('Magnitude of Average Power for turbine')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.tick_params(axis='both', which='major', pad=1)
    # plt.zticks(fontsize=10)
    plt.show()
