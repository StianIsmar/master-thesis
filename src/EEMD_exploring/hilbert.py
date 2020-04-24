import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp


def perform_hilbert(signal,t):
    fs=25600
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1 = fig.add_subplot(212)
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    return amplitude_envelope
    


