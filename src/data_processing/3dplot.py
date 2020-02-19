from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np

import wt_data

print("Loaded wt_data")

import ff_transform

print("Loaded ff_transform")

SENSOR_NAME = 'GbxHssRr;0,0102;m/s2'
BINS = 50

#wt_instance = wt_data.create_wt_data("WTG01", True)
wt_instance = wt_data.load_instance("WTG01", load_minimal=True)

print(f"This is the amount of intervals: {len(wt_instance.ten_second_intervals)}")
y = []
x = []
z = []
i = 0
for interval in wt_instance.ten_second_intervals:
    print(interval.sensor_df['TimeStamp'])

    ts = interval.sensor_df['TimeStamp'].values  # Have this as the y-axis to see how the RMS/frequencies develop

    # y_val = np.array([ts[0]])

    # Have the x axis as frequency!
    # Have the Z axis as amplitude of frequency...
    try:
        vibration_signal = interval.sensor_df[SENSOR_NAME]
    except:
        continue


    y_repeated = np.repeat(i, 50)  # Repeat this y value n times to use as the y value for the corresponding x (frequency) and z (magnitude)
    y.append(y_repeated)
    i = i + 1

    comp_type = 'gearbox'
    fast = ff_transform.FastFourierTransform(vibration_signal, ts, comp_type)
    fft, time, centroid, rms, rms_bins, bin_freq = fast.fft_transform_time(calc_rms_for_bins=True,
                                                                 plot=True,
                                                                 bins=BINS,
                                                                 plot_vertical_lines=True)
    N = fast.s.size
    T = fast.t[1] - fast.t[0]
    f = np.linspace(0, 1 / T, N, )
    f = f[:N // 2]



    # z_amp = np.abs(fft)[:N // 2] * 1 / N
    z.append(rms_bins)

    x.append(bin_freq)
x = np.array(x)
y = np.array(y)
z = np.array(z)

#x = np.array(x[0])
#y = np.array(y[0])
#z = np.array(z[0])

print(f"Length of y: {y.shape}. Len of x: {x.shape}. Len of z; {z.shape}")
# Generate test data


Ny, Nx = len(x), len(x)
# x = np.linspace(0,1,Nx)
# y = np.linspace(0,1,Ny)


# x, y = np.meshgrid(x, y)
# z = (1 - x ** 2) * y ** .5

# Indices for the y-slices and corresponding slice positions along y-axis
# slice_i = [20, 40, 60, 80]
# slice_pos = np.array(slice_i) / Ny


def polygon_under_graph(x, y):
    '''
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph.  Assumes the xlist are in ascending order.
    '''
    return [(x[0], 0.)] + list(zip(x, y)) + [(x[-1], 0.)]


fig = plt.figure()
ax = fig.gca(projection='3d')

verts = []
#for i in slice_i:
    #verts.append(polygon_under_graph(x[i], z[i]))

poly = PolyCollection(verts, facecolors='gray', edgecolors='k')

# add slices to 3d plot
#ax.add_collection3d(poly, zs=slice_pos, zdir='y')

# plot surface between first and last slice as a mesh
#ax.plot_surface(x[slice_i[0]:slice_i[-1]],
                #y[slice_i[0]:slice_i[-1]],
                #z[slice_i[0]:slice_i[-1]],
                #rstride=10, cstride=10, alpha=0, edgecolors='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
