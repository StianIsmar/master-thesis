import numpy as np
import matplotlib.pyplot as plt



'''
Resamples vibration data by using interpolation
Input: 
time_stamps = the x_values, vibration_signal = accelerometer measurements (y_values)
peak_array = time stamp of a shaft revolution, number_of_resample_points = desired number of resampled ponts
Output: 
X_values_round_domain_list = a list of lists of round time (each nested list is one round)
resampled_y_values = list of lists of resampled signal values (each nested list is one round)
all_x_round_domain = one list of round time for all rounds
all_resampled_y_values = one list of all resampled signal values
'''
def linear_interpolation_resampling(time_stamps, vibration_signal, peak_array, number_of_resample_points,
                                    round_plots=0, printing=False, plotting=False):
    # Convert Panda series into numpy arrays for easier data processing
    time_stamps = np.array(time_stamps)
    vibration_signal = np.array(vibration_signal)
    peak_array = np.array(peak_array)

    # Get the index's for each shaft revolution from time_stamps
    peak_indexes = []
    for i, peak in enumerate(peak_array):
        peak_indexes.append(np.where(time_stamps == peak))
    peak_indexes = np.array(peak_indexes).flatten()

    # Extract the every revolution interval, both from time_stamps and vibration_signal
    x_interval = []
    y_interval = []
    for i in range(len(peak_indexes)-1):
        one_x_interval = time_stamps[peak_indexes[i]:peak_indexes[i+1]-1]
        one_y_interval = vibration_signal[peak_indexes[i]:peak_indexes[i + 1] - 1]
        x_interval.append(one_x_interval)
        y_interval.append(one_y_interval)

    # Convert to numpy arrays for easier data processing
    x_interval = np.array(x_interval)
    y_interval = np.array(y_interval)

    resampled_x_values = []
    resampled_y_values = []
    X_values_round_domain_list = []
    for i, one_interval in enumerate(x_interval):
        # Resample the x-coordinates from the beginning of a revolution to its end with the specified number of data points
        start_peak = x_interval[i][0]
        end_peak   = x_interval[i][-1]
        resampled_x = np.linspace(start_peak, end_peak, number_of_resample_points)

        # Resample the vibration data by linear interpolation
        resampled_y = np.interp(resampled_x, x_interval[i], y_interval[i])

        # Transform the x values from time domain to radians domain
        delta_x = resampled_x[1] - resampled_x[0]
        round_start = 2*np.pi * i
        round_end = 2*np.pi * i + 2*np.pi - delta_x
        X_values_round_domain = np.linspace(round_start, round_end, number_of_resample_points)

        # Collect the transformed lists
        resampled_x_values.append(resampled_x)
        resampled_y_values.append(resampled_y)
        X_values_round_domain_list.append(X_values_round_domain)

    resampled_x_values = np.array(resampled_x_values)
    resampled_y_values = np.array(resampled_y_values)
    X_values_round_domain_list = np.array(X_values_round_domain_list)

    # Print various values to check if everything is as it should
    if printing:
        print(f'First original x_interval value: {x_interval[0][0]}')
        print(f'Last  original x_interval value: {x_interval[0][-1]}')
        print(f'Whole original x_interval: {x_interval[0]}')
        print(f'Shape of original x_interval: {x_interval[0].shape}')

        print(f'\nFirst resampled x_interval value: {resampled_x_values[0][0]}')
        print(f'Last  resampled x_interval value: {resampled_x_values[0][-1]}')
        print(f'Shape of resampled x_interval: {resampled_x_values[0].shape}')

        print(f'\nFirst original vibration value: {y_interval[0][0]}')
        print(f'Last  original vibration value: {y_interval[0][-1]}')
        print(f'Shape of first original vibration set: {y_interval[0].shape}')

        print(f'\nFirst resampled vibration value: {resampled_y_values[0][0]}')
        print(f'Last  resampled vibration value: {resampled_y_values[0][-1]}')
        print(f'Shape of first resampled vibration set: {resampled_y_values[0].shape}')

        print(f'\nFirst ronund domain x value: {X_values_round_domain_list[0][0]}')
        print(f'Last  ronund domain x value: {X_values_round_domain_list[0][-1]}')
        print(f'Shape of ronund domain x values: {X_values_round_domain_list[0].shape}')


    if plotting:
        # ------ Plot original signal -------
        x_original = []
        y_original = []
        for i in range(round_plots):
            x_original = np.append(x_original, x_interval[i])
            y_original = np.append(y_original, y_interval[i])
        original_vertical_lines = peak_array[0:round_plots+1]

        plt.figure(figsize=(20, 10))
        plt.plot(x_original, y_original, c='b', linewidth=0.1)
        plt.title(f'Original Vibration Data. Number of Data Points: {x_original.shape[0]}', fontsize=20)
        plt.xlabel('Time (in s)', fontsize=16)
        plt.ylabel('Vibration amplitude (in m/s2)', fontsize=16)
        for i, round_value in enumerate(original_vertical_lines):
            plt.axvline(x=round_value, c='r', linewidth=0.3)
        plt.margins(0)
        plt.show()


        # ------ Plot resampled signal ------
        x_resampled = []
        y_resampled = []
        resampled_vertical_lines = []
        for i in range(round_plots):
            x_resampled = np.append(x_resampled, X_values_round_domain_list[i])
            y_resampled = np.append(y_resampled, resampled_y_values[i])
            resampled_vertical_lines.append(X_values_round_domain_list[i][0])

        plt.figure(figsize=(20, 10))
        plt.plot(x_resampled, y_resampled, c='b', linewidth=0.1)
        plt.title(f'Resampled Vibration Data. Number of Data Points: {x_resampled.shape[0]}', fontsize=20)
        plt.xlabel('Rounds (in radians)', fontsize=16)
        plt.ylabel('Vibration amplitude (in m/s2', fontsize=16)
        for i, round_value in enumerate(resampled_vertical_lines):
            plt.axvline(x=round_value, c='r', linewidth=0.3)
        plt.margins(0)
        plt.show()


    all_resampled_y_values = np.array([])
    all_x_round_domain = np.array([])
    for i, y_val in enumerate(resampled_y_values):
        all_resampled_y_values = np.append(all_resampled_y_values, np.array(y_val))
        all_x_round_domain = np.append(all_x_round_domain, np.array(X_values_round_domain_list[i]))


    return X_values_round_domain_list, resampled_y_values, all_x_round_domain, all_resampled_y_values