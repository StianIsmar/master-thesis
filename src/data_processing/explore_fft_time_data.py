#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sys, os

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


# In[ ]:


## 2D plotting function
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


# In[ ]:


## Creating a 3d plot using PolyCollection (X: Frequency, Y: Interval number, Z: RMS)
'''
    Create a 3d-plot using the poly collection module from matplotlib.
'''
from matplotlib import cm, pyplot as plt
import math


# In[ ]:


import matplotlib
import matplotlib.cm as cmx
from sklearn.preprocessing import minmax_scale
'''
    Create a 3d-plot using the poly collection module from matplotlib.
'''
from matplotlib import cm, pyplot as plt
import math
def print3d_with_poly_collection(t,remove_indexes_01,x,y,z,color_alt,average_powers, cm_style='Blues',filter = False):

    # Delete from the avg_powers and average_rot_speed array
    for i, index in enumerate(remove_indexes_01):
        y = np.delete(y, [index], axis=0)
        x = np.delete(x, [index], axis=0)
        z = np.delete(z, [index], axis=0)
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
        #print(f"MIN: {min(norm)}. MAX: {max(norm)}")
        col = [cmap(x) for x in scaled]
        # print(col)
        poly = PolyCollection(verts,facecolors=col)
        # poly.set_cmap(cmap)
        # poly.set_norm(norm)
        # poly.update_scalarmappable()
        # poly.set_facecolors('k')
        # poly.set_cmap(cmap)
        # poly.set_norm(norm)
    else:
        poly = PolyCollection(verts)
    poly.set_alpha(0.7)

    # The zdir keyword makes it plot the "z" vertex dimension (radius)
    # along the y axis. The zs keyword sets each polygon at the
    # correct radius value.
    ax.add_collection3d(poly, zs=rad_data, zdir='y')
    ax.set_xlim3d(freq_data.min(), freq_data.max())
    ticks = np.arange(0, freq_data.max(), 250)
    ax.set_xticks(ticks)
    
    # ax.set_xticks(np.arange(0,freq_data.max(),10))
    ax.set_xlabel('Frequency [Hz]',labelpad=10)
    ax.set_ylim3d(rad_data.min(), rad_data.max())


    ax.set_ylabel('Interval number',labelpad=10)
    # ax.set_zlim3d(amp_data.min(), amp_data.max())
    ax.set_zlim3d(amp_data.min(), 8)
    ax.set_zlabel('RMS amplitude')

    # Colourbar
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
    
'''print3d_with_poly_collection("1",remove_indexes_wt01,
                             freqs_wt01,
                             interval_nums_wt01, 
                             rms_amplitudes_wt01,
                             'color_alt4',
                             avg_powers_filtered_wt01,
                             'Blues',
                             False)
                             
'''


# In[ ]:


### Remove spikes that seem unnaturally large (by inspection). From the 3D plots,
### one can see that the spikes show up "randomly". They are therefore filtered away
def filter_RMS_spikes(x,y,z):
    for i, array in enumerate(z):
        for j, rms_val in enumerate(array):
            if rms_val > 10:
                # divide it by 2
                z[i][j] = rms_val/2
                print(f"Previous value: {rms_val}. New value: {z[i][j]}")
                
    return z


# In[ ]:


## Creating surface 3d plots with axes3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def print3d_with_axes3d(x,y,z,cm_style='Blues'):
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    
    X = X.T
    Y = Y.T
    Z = Z.T
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cmap = cm.get_cmap(cm_style)
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=False)
    plt.title("Frequency development over intervals for WT")
    plt.show()


# In[ ]:


## Filtering away the noise (some of the avg. power values were at -1*10^37)
'''
    Takes in the amount of bins to be plotted (bin_amount),
    the average powers list for each interval, and
    the bin_rms_values
'''
def scatter_plot_rms_avg_power(bin_list, avg_powers, bin_rms_values,wt_num,every_bin_range=256):
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
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*every_bin_range},{bin_list[-1]*every_bin_range}] Hz")
    
    
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


# In[ ]:


## Plotting RMS (y-axis) agains ROT speed (x-axis)
def scatter_plot_rms_rot_speed(bin_list, wt_num,two_d_plot_tw,avg_rot_speeds,every_bin_range=256): # Takes in how many bin you want to have studied
    try:
        wt_num = wt_num.split('0')[1]
        
    except:
        print("wt_num is not the wrong format")
    

        
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
    plt.title(f"RMS and RPM correlation for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*every_bin_range},{bin_list[-1]*every_bin_range}] Hz")

    plt.ylim(0,max_rms_val+1)
    plt.xlim(0,1650)
    plt.xlabel("Mean RPM")
    plt.ylabel("RMS")
    plt.show()


# In[ ]:


import warnings
warnings.simplefilter('ignore')

def scatter_plot_rms_rot_speed_time(bin_list, wt_num, two_d_plot_tw, avg_rot_speeds, freq_in_bins = 256): # Takes in how many bin you want to have studied
    try:
        wt_num = wt_num.split('0')[1]
    except:
        print("wt_num is not the wrong format")
        
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
    ax.set_title(f"RMS and RPM correlation for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*freq_in_bins},{bin_list[-1]*freq_in_bins}] Hz")

    ax.set_ylim(0,max_rms_val+0.001)
    ax.set_xlim(0,np.ceil(len(two_d_plot_tw[bin_num])))
    
    ax.set_xlabel("Interval Number")
    ax1.set_xlabel("Interval Number")

    ax.set_ylabel("RMS / Avg. Speed")
    ax1.set_ylabel("RMS / Avg. Speed")

    plt.show()


# In[ ]:


## Plotting RMS (y-axis) agains Average power (x-axis)
'''
    Takes in the amount of bins to be plotted (bin_amount),
    the average powers list for each interval, and
    the bin_rms_values
'''
def scatter_plot_rms_avg_power(bin_list, avg_powers, bin_rms_values,wt_num,every_bin_range=256):
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
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{bin_list[0]*every_bin_range},{bin_list[-1]*every_bin_range}] Hz")
    
    
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


# In[ ]:





# In[ ]:





# In[ ]:




