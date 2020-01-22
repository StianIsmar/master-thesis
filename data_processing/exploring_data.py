'''
import process_data

interval = process_data.interval


print(interval.date)

'''
import pickle

content = pickle.load(open('saved_dfs.p', 'rb'))

print(content['op_df'])
print()
print(content['sensor_data_df'])
