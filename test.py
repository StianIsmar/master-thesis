import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos

fig = plt.figure()

# preferred method for creating 3d axis
ax = fig.add_subplot(111, projection='3d')
r = 10
c = 50
t = np.linspace(0, 5000, 100)

# parametric equation of a helix
x = r * cos(t)
y = r * sin(t)
z = c * t

ax.scatter(x, y, z, zdir='z', lw=2, c='g', marker='^')
