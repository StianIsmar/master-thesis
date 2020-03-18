from scipy.signal import butter, lfilter,filtfilt

# bandpass
def butter_hp(lowcut, fs, order=5):
    nyq = 0.5 * fs
    w = lowcut / nyq
    b, a = butter(order, w, btype='highpass')
    return b, a

def butter_hp_filter(data, lowcut, fs, order=5):
    b, a = butter_hp(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#lp
def butter_lp(highcut, fs, order=5):
    nyq = 0.5 * fs
    w = highcut / nyq
    b, a = butter(order, w, btype='lowpass')
    return b, a

def butter_lp_filter(data, highcut, fs, order=5):
    b, a = butter_lp(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y