#!!pip install pyuff
import pyuff
import pprint
import pandas as pd
import seaborn as sbs

print ("Start read")
pp = pprint.PrettyPrinter(indent=4)

# uff_file = pyuff.UFF('/Volumes/Harddriver/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')
uff_file = pyuff.UFF('/Users/stian/Desktop/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')

def get_data(uff):
    # Get which datasets are written in the file
    # print((uff_file.get_set_types()))
    print(len(uff_file.get_set_types()))
    data = uff_file.read_sets()
    return data


data = get_data(uff_file)
print(len(data))
print("DATA3KEYS")

pp.pprint(data[0].keys())


def get_dataframe(d):

    df_other = pd.DataFrame()
    df_vib = pd.DataFrame()

    for i in range(len(d)):
        x = d[i]['x']

        if (len(x) > 0 ):
            # Add this to the large DataFrame
            id_3 = (d[i]['id3']) # some kind of time..
            id_4 = (d[i]['id4']) # Which turbine we are dealing with
            id_5 = (d[i]['id5']) # Parameter name and unit
            print(id_5)
            TimeStamp = x
            data = (d[i]['data'])

            df_vib["TimeStamp"] = x

            df_vib[id_5] = data

        else:
            # Add it to the other dataframe
            print("Add to other...")

    return df_vib


df_vib = get_dataframe(data)
print("df_vib")
print(df_vib)


def plot_df(dataframe):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    print(plot_df)
    for y in range(dataframe.shape[1]):

        sns.set_style("darkgrid")

        plt.plot(dataframe["TimeStamp"], dataframe.iloc[ : , y ],linewidth=0.1)

        # ax1 = plt.add_subplot(1,1,1,axisbg='grey')
        # ax1.plot(x, y, 'c', linewidth=3.3)
        plt.show()



plot_df(df_vib)

# df = pd.DataFrame(columns=data[1]['id5'])
# print(df)
# print(df.columns)








'''for i in range(len(uff_file.get_set_types())):
    print(i)
    print(" ")

    # Examine the header of each set
    pp.pprint(uff_file.read_sets(i))

    print(" ")
'''

