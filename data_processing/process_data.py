import pyuff
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seabon as sns; sns.set()


print ("Start read")
pp = pprint.PrettyPrinter(indent=4)


uff_file = pyuff.UFF('/Volumes/OsvikExtra/VibrationData/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')

data = uff_file.read_sets()



time    = data[0]['id3']
turbine = data[0]['id4']

x             = []
single__name  = []
multi_name    = []
single_info   = []
multi_info    = []

for i in range(len(data)):

    name = data[i]['id5']
    info = (data[i]['data'])


    if len(info) > 0:
        multi_name.append(name)
        multi_info.append(info)

        if len(x) == 0:
            x.append(data[i]['x'])
            print('x', x)
            print('info', info)

multi_info.append(x[0])
multi_name.append('TimeStamp')
multi_info_np = np.array(multi_info)

#single_pd = pd.DataFrame(single_info, columns=single__name)
multi_pd = pd.DataFrame(multi_info_np.T, columns=multi_name)



print(name)
print(time)
print(turbine)
print("X", x)
print('info', info)
print()
pp.pprint(data[2])
print()
print('length of multi_name', len(multi_name))
print()
print('length of multi_info', len(multi_info))
print(multi_info_np.shape)
print()
print(len(x))
print(multi_pd)


for i in range(len(multi_name)):
    plt.plot(multi_pd['TimeStamp'][0:12500], multi_pd[multi_name[i]][0:12500], linewidth=0.1)
    plt.title(multi_name[i] + ' VS time')
    plt.ylabel(multi_name[i])
    plt.xlabel('Time')
    plt.show()



