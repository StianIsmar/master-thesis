import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sys, os,os.path
from numpy import savez_compressed
from numpy import load

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

# %autoreload
import wt_data
import ff_transform
import explore_fft_time_data
import envelope
import build_dataset

'''
component: 'gearbox'
wt: 'wt01', 'wt02','wt03', or 'wt04'
'''
def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]


def load_raw(component, wt):
	if component == 'gearbox':
		sensor = 'GbxHssRr;0,0102;m/s2'
		path = f'/Volumes/OsvikExtra/signal_data/raw_data/gearbox/{wt}'
		filepath_vib = path + f'/raw_gearbox_sig_{wt}.npz'
		filepath_times = path + f'/time_stamps_{wt}.npz'
		filepath_op_data = path + f'/op_data_{wt}.npz'
		filepath_peak_array = path + f'/peak_array_{wt}.npz'
		if os.path.isfile(filepath_vib) and os.path.isfile(filepath_times) and os.path.isfile(filepath_peak_array) and os.path.isfile(filepath_op_data):
			print("File exist")

			# just load the data:

			# Load compressed data
			vib_signal = load(filepath_vib)
			vib_signal = vig_signal['arr_0']

			times = load(filepath_times)
			times = times['arr_0']

			op_data_intervals = pd.read_csv(filepath_op_data, compression='gzip')


			peak_array = load(filepath_peak_array, allow_pickle=True)
			peak_array = peak_array['arr_0']

			return vib_signal, times, op_data_intervals, peak_array
		
		else:
			print("Building the data")
			wt_name = wt.upper()
			wt_name = insert_str(wt_name,'G',2)
			sig, times, intervals_op_data, peak_array = build_dataset.load_wt_high_freq_analysis(wt_name, sensor)
			print("Starting to save the data.")

			# Saving vibrations
			if not (os.path.isfile(filepath_vib)):
				savez_compressed(filepath_vib, sig)

			# Saving timestamps
			if not (os.path.isfile(filepath_times)):
				savez_compressed(filepath_times, times)

			# Saving operating data
			if not (os.path.isfile(filepath_op_data)):
				intervals_op_data.to_csv(filepath_op_data, compression='gzip')

			if not (os.path.isfile(filepath_peak_array)):
				savez_compressed(filepath_peak_array, peak_array)
			
			print("Done saving the data")

			del sig, times, intervals_op_data, peak_array

			load_raw(component, wt)

load_raw('gearbox','wt01')
#load_raw('gearbox','wt02')
#load_raw('gearbox','wt03')
# load_raw('gearbox','wt04')


