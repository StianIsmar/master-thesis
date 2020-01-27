'''
import process_data

interval = process_data.interval


print(interval.date)

'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import process_data
import wt_data

def plot_data(dataframe, colName, plot_all=False):
    x_values = dataframe['TimeStamp']
    dataframe = dataframe.drop(columns=['TimeStamp'])
    sensor_name = list(dataframe.columns.values)

    if plot_all:
        for i in range(len(sensor_name)):
            if sensor_name[i] == "LssShf;1;V":  # Plot the whole 10 sec for this sensor.
                plt.plot(x_values, dataframe[sensor_name[i]], linewidth=0.1)
            else:
                plt.plot(x_values[0:12500], dataframe[sensor_name[i]][0:12500], linewidth=0.1)

            plt.title(sensor_name[i] + ' VS time')
            plt.ylabel(sensor_name[i])
            plt.xlabel('Time')
            plt.show()

    else:
        plt.plot(x_values, dataframe[colName], linewidth=0.1)
        plt.title(colName + ' VS time')
        plt.ylabel(colName)
        plt.xlabel('Time')
        plt.show()



'''
content = pickle.load(open('saved_dfs.p', 'rb'))

op_df = content['op_df']
sensor_data_df = content['sensor_data_df']

print(sensor_data_df.columns.values)

plot_data(sensor_data_df, 'Speed Sensor;1;V')

plot_data(sensor_data_df, 'LssShf;1;V')

low_rot_speed = calc_avg_speed(sensor_data_df, 'LssShf;1;V')
high_rot_seed = calc_avg_speed(sensor_data_df, 'Speed Sensor;1;V')

print('Low rotational speed is:   ', low_rot_speed)
print('High rotational speed is:  ', high_rot_seed)
print(op_df)
print(op_df.shape)

op_df.insert(len(op_df.columns.values), "LowSpeed:rps", low_rot_speed)
op_df.insert(len(op_df.columns.values), "HighSpeed:rps", high_rot_seed)

print(op_df)
print(op_df.shape)
'''






#instance = process_data.TenSecondInterval()

#instance = process_data.load_instance()
#print(instance.date)



# --------- TO CREATE WT INSTANCES --------------

#wt_instance_1 = wt_data.create_wt_data("WTG01")
#wt_instance_2 = wt_data.create_wt_data("WTG02")
#wt_instance_3 = wt_data.create_wt_data("WTG03")
#wt_instance_4 = wt_data.create_wt_data("WTG04")


# ---------  TO LOAD WT INSTANCES --------------

#wt_instance_1 = wt_data.load_instance("WTG01")
#wt_instance_2 = wt_data.load_instance("WTG02")
#wt_instance_3 = wt_data.load_instance("WTG03")
#wt_instance_4 = wt_data.load_instance("WTG04")



