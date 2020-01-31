import wt_data
import fft_transform


wt_instance = wt_data.load_instance("WTG01",load_minimal=True)
intervals = wt_instance.ten_second_intervals
for i, interval in enumerate(intervals):
    # Check if the interval is relevant with Mortens code
    time_stamps = interval.sensor_df['TimeStamp']
    vibration_signal = interval.sensor_df['GnNDe;0,0102;m/s2']
    #

fast = fft_transform.FastFourierTransform()
# fast.fft_transform()

