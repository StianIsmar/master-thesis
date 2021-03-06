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
def print3d_with_poly_collection(turbine_number,sensor_name, remove_indexes,x,y,z,color_alt,average_powers, cm_style='Blues',filter = False):

    # Delete from the avg_powers and average_rot_speed array
    for i, index in enumerate(remove_indexes):
        y = np.delete(y, [index], axis=0)
        x = np.delete(x, [index], axis=0)
        z = np.delete(z, [index], axis=0)
    if filter == True:
        z = filter_RMS_spikes(x,y,z)
    
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"FFT development over time for WT {turbine_number}",pad=20)
    
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
        # cmap="coolwarm"
        cmap="copper"
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
    ax.set_zlabel(f' {sensor_name} RMS amplitude')

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
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{int(bin_list[0]*every_bin_range)},{int(bin_list[-1]*every_bin_range)}] Hz")
    
    
def filter_data(avg_powers, RMS_per_bin, average_rpm, wind_speeds, rms_amplitudes): # filter the data based on the really avg low power values
    
    print("Old min power value: ", np.min(avg_powers))

    return_dict = {
    "avg_powers_filtered": None,
    "RMS_per_bin_filtered": None,
    "average_rpm_filtered": None,
    "remove_indexes": None,
    "wind_speeds_filtered": None,
    'rms_amplitudes': None
    }

    avg_powers_filtered = avg_powers
    RMS_per_bin_filtered = RMS_per_bin
    average_rpm_filtered = average_rpm
    wind_speeds_filtered = wind_speeds

    indexes = []

    # Find out where these extreme values lie in the average powers array
    for i, val in enumerate(avg_powers_filtered):
        if val <= 0:
            indexes.append(i)
    
    for j, val in enumerate(wind_speeds):
        if val <=0 and (i not in indexes):
            indexes.append(j)

    indexes.sort()
    indexes.reverse() # reverse list in order to delete the low powers
    if (len(indexes) == 0):
        print("Already ran.. ")
        return return_dict
    else:
        # Delete from the avg_powers and average_rot_speed array
        for i, index in enumerate(indexes):
            del avg_powers_filtered[index]
            del average_rpm_filtered[index]
            del wind_speeds_filtered[index]
            rms_amplitudes = np.delete(rms_amplitudes, [index], axis=0)

        # Delete from the RMS bin lists
        for index in indexes:
            for i, rms_bin in enumerate(RMS_per_bin_filtered):
                del rms_bin[index]

        print("New min power value", np.min(avg_powers_filtered))
        remove_indexes = indexes
        
        return_dict["avg_powers_filtered"] = avg_powers_filtered
        return_dict["RMS_per_bin_filtered"] = RMS_per_bin_filtered
        return_dict["average_rpm_filtered"] = average_rpm_filtered
        return_dict["remove_indexes"] = remove_indexes
        return_dict["wind_speeds_filtered"] = wind_speeds_filtered
        return_dict['rms_amplitudes'] = rms_amplitudes

        return return_dict


# In[ ]:


## Plotting RMS (y-axis) agains ROT speed (x-axis)
def scatter_plot_rms_rot_speed(bin_list, wt_num,two_d_plot_tw,avg_rot_speeds,every_bin_range=256, xlab="Mean RPM",ylab="RMS",x_axis_variable = "rot_speed"): # Takes in how many bin you want to have studied
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

    if x_axis_variable == "rot_speed":
        plt.title(f"RMS and RPM correlation for WT {wt_num}" "\n" f"Frequency range: [{int(bin_list[0]*every_bin_range)},{int(bin_list[-1]*every_bin_range)}] Hz")
    elif x_axis_variable == "wind_speed":
        plt.title(f"RMS and wind speed correlation for WT {wt_num}" "\n" f"Frequency range: [{int(bin_list[0]*every_bin_range)},{int(bin_list[-1]*every_bin_range)}] Hz")

    plt.ylim(0,max_rms_val*1.01)
    plt.xlim(0,max(avg_rot_speeds)*1.01)
    plt.xlabel(f"{xlab}")
    plt.ylabel(f"{ylab}")
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

    ax.set_ylim(0,max_rms_val*1.01)
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
    plt.title(f"RMS and Average Power for WT {wt_num}" "\n" f"Frequency range: [{int(bin_list[0]*every_bin_range)},{int(bin_list[-1]*every_bin_range)}] Hz")
    

def rms_wind_speed_scatter(filtered_rms_amplitudes_wt, filtered_wind_speeds):
    first_row = filtered_rms_amplitudes_wt



# In[ ]:
'''
Params:
    - Turbine number: "1", "2", "3", or "4"
    - Sensor name: For the plots
    - x (Frequencies): Matrix with frequencies, bin indexes (for the x axis)
    - y (Intervals): Matrix with interval indexes [[0,0,0,0..][1,1,1,1,..][2,2,2,2,..].. [414,414,414,..]]
    - z (Amplitudes): Matrix with frequency amplitudes [[],[],[]]

    EXAMPLE CALL:
    
    x = np.array([[1,50,100],[1,50,100],[1,50,100]])
    y = np.array([[1,1,1],[2,2,2],[2,2,2]])
    z = np.array([[1,2,1],[1,2,1],[1,2,1]])
    average_powers = [200,100,100]
    turbine_number = "1"
    sensor_name = "My sensor"
    plot_title = "RMS amplitude over time for 'sensor name'"
    z_max_plot = 20
    colorbar_val_min = 10
    colorbar_val_max = 1337
    x_axis_title = "Order [X]"
    cm_style = "copper"

    print3d_modular(x,y,z,turbine_number,sensor_name,plot_title, z_max_plot, colorbar_val_min, colorbar_val_max, x_axis_title, average_powers, cm_style=cm_style)
'''
def print3d_modular(x,y,z,turbine_number,sensor_name,plot_title, z_max_plot, colorbar_val_min, colorbar_val_max, x_axis_title, average_powers, cm_style='Blues'):
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{plot_title} {turbine_number}",pad=20)
    
    # Get the numpy arrays on the correct shape
    freq_data = x.T
    amp_data = z.T
    rad_data = np.linspace(0,amp_data.shape[1]-1,amp_data.shape[1])
    
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
    poly = PolyCollection(verts,facecolors=col)

    
    poly.set_alpha(0.7)

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

# In[ ]:

# In[ ]:




