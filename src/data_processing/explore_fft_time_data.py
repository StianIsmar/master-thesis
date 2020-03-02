import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sys, os

import matplotlib
import matplotlib.cm as cmx
from sklearn.preprocessing import minmax_scale

from matplotlib import cm, pyplot as plt
import math

ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
print("ROOT", ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/utils/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)

import functions as f

module_path = os.path.abspath(os.path.join(ROOT_PATH+"/data_processing/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)
    
import wt_data

import ff_transform


# Plotting 2d bins
def plot_2d_bins(bin_values):
    num_intervals = len(bin_values[0])
    x_news = []
    y_news = []
    for j, binn in enumerate(bin_values):

        y = binn
        x = np.arange(0,len(binn))
        
        z = np.polyfit(x, y, 3)
        f = np.poly1d(z)
        plt.plot(x,y)
        x_new = np.linspace(x[0], x[-1], 50)
        y_new = f(x_new)

        x_news.append(x_new)
        y_news.append(y_new)
    fig = plt.figure(figsize = (15,10))
    for i,x in enumerate(x_news):
        plt.plot(x,y_news[i], label=1)
        x_post = i %  50
        plt.ylim(top = np.max(y_news) + 0.2)

    plt.margins(0)
    fig.tight_layout() 

    plt.show()


'''
    Create a 3d-plot using the poly collection module from matplotlib.
'''

def print3d_with_poly_collection(t,remove_indexes_01,x,y,z,color_alt,average_powers, cm_style='Blues',filter = False):

    for i, index in enumerate(remove_indexes_01):
        y = np.delete(y, [index], axis=0)
        x = np.delete(x, [index], axis=0)
        z = np.delete(z, [index], axis=0)
            
    print("len(x): ", len(x))
    print("len(y): ", len(y))
    print("len(z): ", len(z))

    print(len(average_powers))
    if filter == True:
        z = filter_RMS_spikes(x,y,z)
    
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"FFT development over time for WT {t}",pad=20)
    
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
    
    if color_alt == 'color_alt4':
        cmap="Blues"
        cmap = cmx.get_cmap(cmap)
        scaled = minmax_scale(average_powers)
        col = [cmap(x) for x in scaled]
        poly = PolyCollection(verts,facecolors=col)

    else:
        poly = PolyCollection(verts)
    poly.set_alpha(0.7)

    # The zdir keyword makes it plot the "z" vertex dimension (radius)
    # along the y axis. The zs keyword sets each polygon at the
    # correct radius value.
    ax.add_collection3d(poly, zs=rad_data, zdir='y')
    ax.set_xlim3d(freq_data.min(), freq_data.max())
    ax.set_xlabel('Frequency [Hz]',labelpad=10)
    ax.set_ylim3d(rad_data.min(), rad_data.max())
    ax.set_ylabel('Interval number',labelpad=10)
    # ax.set_zlim3d(amp_data.min(), amp_data.max())
    ax.set_zlim3d(amp_data.min(), 8)
    ax.set_zlabel('RMS amplitude')

    # Colourbar
    print(max(average_powers)) # 3314.455810547
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    mn=int(np.floor(min(average_powers)))  
    mx=int(np.ceil(max(average_powers)))
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
    
'''
    Takes in the amount of bins to be plotted (bin_amount),
    the average powers list for each interval, and
    the bin_rms_values
'''
def scatter_plot_rms_avg_power(bin_list, avg_powers, bin_rms_values,wt_num):
    try:
        wt_num = wt_num.split('0')[1]
        
    except:
        print("wt_num is not the wrong format")
    
    rows = int(np.floor(len(bin_list)/3))
    check = len(bin_list) % 3
    if check > 0:
        rows += 1
        
    fig = plt.figure()    
    
    for bin_num in (bin_list):
        if len(bin_rms_values[bin_num]) == len(avg_powers):
            ax = fig.add_subplot(1,1,1)
            ax.scatter(avg_powers,bin_rms_values[bin_num])
        else:
            print(f"The shape is not the same. {len(bin_rms_values[bin_num])} =! {len(avg_powers)}'")
            
    
    len(bin_list)
    legend_labels = ['Bin ' + str(elem) for i, elem in enumerate(bin_list)]
    ax.legend(labels=legend_labels, loc='upper left',
             markerscale=3.,fontsize=10)
    plt.xlabel("Average power [kW]")
    plt.margins(0)
    print((max(avg_powers)))
    plt.xticks(np.arange(0, int(math.ceil(max(avg_powers) / 100.0)) * 100, 200))

    plt.ylabel("RMS")
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*256},{bin_list[-1]*256}] Hz")
    
    
def filter_data(avg_powers, RMS_per_bin, average_rpm): # filter the data based on the really avg low power values
    
    print("Old min power value: ", np.min(avg_powers))

    avg_powers_filtered = avg_powers
    RMS_per_bin_filtered = RMS_per_bin
    average_rpm_filtered = average_rpm

    indexes = []

    # Find out where these extreme values lie in the average powers array
    for i, val in enumerate(avg_powers_filtered):
        if val <= 0:
            indexes.append(i)
    indexes.reverse() # reverse list in order to delete the low powers
    
    if (len(indexes) == 0):
        print("Already ran.. ")
    else:
        # Delete from the avg_powers and average_rot_speed array
        for i, index in enumerate(indexes):
            del avg_powers_filtered[index]
            del average_rpm_filtered[index]

        # Delete from the RMS bin lists
        for index in indexes:
            for i, rms_bin in enumerate(RMS_per_bin_filtered):
                del rms_bin[index]

        print("New min power value", np.min(avg_powers_filtered))
        remove_indexes = indexes
        return avg_powers_filtered, RMS_per_bin_filtered, average_rpm_filtered, remove_indexes


# Plotting RMS (y-axis) agains ROT speed (x-axis)
def scatter_plot_rms_rot_speed(bin_list, wt_num): # Takes in how many bin you want to have studied
    try:
        wt_num = wt_num.split('0')[1]
        
    except:
        print("wt_num is not the wrong format")
    
    if wt_num == "1":
        two_d_plot_tw = two_d_plot_tw01
        avg_rot_speeds = avg_rot_speeds1
    if wt_num == "2":
        two_d_plot_tw = two_d_plot_tw02
        avg_rot_speeds = avg_rot_speeds2
    if wt_num == "3":
        two_d_plot_tw = two_d_plot_tw03
        avg_rot_speeds = avg_rot_speeds3
    if wt_num == "4":
        two_d_plot_tw = two_d_plot_tw04
        avg_rot_speeds = avg_rot_speeds4
        
    fig = plt.figure()
    
    max_rms_val = 0
    
    for bin_num in bin_list:
        
        if len(two_d_plot_tw[bin_num]) == len(avg_rot_speeds):
            ax = fig.add_subplot(1,1,1)
            ax.scatter(avg_rot_speeds,two_d_plot_tw[bin_num])
            check_max = np.max(two_d_plot_tw[bin_num])
            max_rms_val = check_max if max_rms_val < check_max else max_rms_val
        else:
            print(f"The shape is not the same. {len(two_d_plot_tw)} =! {len(avg_rot_speeds)}'")
    legend_labels = ['Bin ' + str(elem) for i, elem in enumerate(bin_list)]
    ax.legend(labels=legend_labels,loc='upper left',
             markerscale=2.,fontsize=10)
    plt.margins(0)
    plt.title(f"RMS and RPM correlation for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*256},{bin_list[-1]*256}] Hz")

    plt.ylim(0,max_rms_val+1)
    plt.xlim(0,1650)
    plt.xlabel("Mean RPM")
    plt.ylabel("RMS")
    plt.show()


import warnings
warnings.simplefilter('ignore')

def scatter_plot_rms_rot_speed_time(bin_list, wt_num): # Takes in how many bin you want to have studied
    try:
        wt_num = wt_num.split('0')[1]
    except:
        print("wt_num is not the wrong format")
    
    if wt_num == "1":
        two_d_plot_tw = two_d_plot_tw01
        avg_rot_speeds = avg_rot_speeds1
    if wt_num == "2":
        two_d_plot_tw = two_d_plot_tw02
        avg_rot_speeds = avg_rot_speeds2
    if wt_num == "3":
        two_d_plot_tw = two_d_plot_tw03
        avg_rot_speeds = avg_rot_speeds3
    if wt_num == "4":
        two_d_plot_tw = two_d_plot_tw04
        avg_rot_speeds = avg_rot_speeds4
        
    fig = plt.figure()
    fig1 = plt.figure()
    
    max_rms_val = 0
    for bin_num in bin_list:
        
        if len(two_d_plot_tw[bin_num]) == len(avg_rot_speeds):
            ax = fig.add_subplot(1,1,1)
            ax1 = fig1.add_subplot(1,1,1)

            standardise_rms_rot_speed = [two_d_plot_tw[bin_num][i] / avg_rot_speeds[i] for i in range(len(avg_rot_speeds)) ]
            
            z = np.polyfit(np.arange(0,len(two_d_plot_tw[bin_num])), standardise_rms_rot_speed,1)
            f = np.poly1d(z)
            y_line =  f(np.arange(0,len(two_d_plot_tw[bin_num])))
            
            ax1.plot(y_line)
            
            ax.scatter( np.arange( 0,len(two_d_plot_tw[bin_num])),standardise_rms_rot_speed  )
            
            x = np.arange(0,len(two_d_plot_tw[bin_num]))
            x = [x[0],x[-1]]
            y_line = [y_line[0], y_line[-1]] # These can not be correct? look into this 
            ax1.scatter(x, y_line)
            
            check_max = np.max(standardise_rms_rot_speed)
            max_rms_val = check_max if max_rms_val < check_max else max_rms_val
        else:
            print(f"The shape is not the same. {len(two_d_plot_tw)} =! {len(avg_rot_speeds)}'")
    
    legend_labels = ['Bin ' + str(elem) for i, elem in enumerate(bin_list)]
    ax.legend(labels=legend_labels,loc='upper left',
             markerscale=2.,fontsize=10)
    
    ax1.legend(labels=legend_labels,loc='upper left',
             markerscale=2.,fontsize=10)
    plt.margins(0)
    ax.set_title(f"RMS and RPM correlation for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*256},{bin_list[-1]*256}] Hz")

    ax.set_ylim(0,max_rms_val+0.001)
    ax.set_xlim(0,np.ceil(len(two_d_plot_tw[bin_num])))
    
    ax.set_xlabel("Interval Number")
    ax1.set_xlabel("Interval Number")

    ax.set_ylabel("RMS / Avg. Speed")
    ax1.set_ylabel("RMS / Avg. Speed")

    plt.show()


    ## Plotting RMS (y-axis) agains Average power (x-axis)

    '''
    Takes in the amount of bins to be plotted (bin_amount),
    the average powers list for each interval, and
    the bin_rms_values
'''
def scatter_plot_rms_avg_power(bin_list, avg_powers, bin_rms_values,wt_num):
    try:
        wt_num = wt_num.split('0')[1]
        
    except:
        print("wt_num is not the wrong format")
    
    rows = int(np.floor(len(bin_list)/3))
    check = len(bin_list) % 3
    if check > 0:
        rows += 1
        
    fig = plt.figure()    
    
    for bin_num in (bin_list):
        if len(bin_rms_values[bin_num]) == len(avg_powers):
            ax = fig.add_subplot(1,1,1)
            ax.scatter(avg_powers,bin_rms_values[bin_num])
        else:
            print(f"The shape is not the same. {len(bin_rms_values[bin_num])} =! {len(avg_powers)}'")
            
    
    len(bin_list)
    legend_labels = ['Bin ' + str(elem) for i, elem in enumerate(bin_list)]
    ax.legend(labels=legend_labels, loc='upper left',
             markerscale=3.,fontsize=10)
    plt.xlabel("Average power [kW]")
    plt.margins(0)
    print((max(avg_powers)))
    plt.xticks(np.arange(0, int(math.ceil(max(avg_powers) / 100.0)) * 100, 200))

    plt.ylabel("RMS")
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*256},{bin_list[-1]*256}] Hz")
    
    
def filter_data(avg_powers, RMS_per_bin, average_rpm): # filter the data based on the really avg low power values
    
    print("Old min power value: ", np.min(avg_powers))

    avg_powers_filtered = avg_powers
    RMS_per_bin_filtered = RMS_per_bin
    average_rpm_filtered = average_rpm

    indexes = []

    # Find out where these extreme values lie in the average powers array
    for i, val in enumerate(avg_powers_filtered):
        if val <= 0:
            indexes.append(i)
    indexes.reverse() # reverse list in order to delete the low powers
    
    if (len(indexes) == 0):
        print("Already ran.. ")
    else:
        # Delete from the avg_powers and average_rot_speed array
        for i, index in enumerate(indexes):
            del avg_powers_filtered[index]
            del average_rpm_filtered[index]

        # Delete from the RMS bin lists
        for index in indexes:
            for i, rms_bin in enumerate(RMS_per_bin_filtered):
                del rms_bin[index]

        print("New min power value", np.min(avg_powers_filtered))
        remove_indexes = indexes
        return avg_powers_filtered, RMS_per_bin_filtered, average_rpm_filtered, remove_indexes

