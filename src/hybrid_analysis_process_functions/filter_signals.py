import numpy as np

import filters

'''
INPUT: 
vibration_signal:  array of amplitude of vibration signal
low_cut:           float/int lower frequency limit. Frequencies smaller than low_cut are filtered out
time_duration:     float - duration of vibration signal
OUTPUT:
filtered_signal:   array - amplitude of filtered vibration signal
'''
def do_high_pass_filter(vibration_signal, low_cut, time_duration=10.2399609375):
    fs = len(vibration_signal) / time_duration
    filtered_signal = filters.butter_hp_filter(vibration_signal, low_cut, fs)
    return filtered_signal