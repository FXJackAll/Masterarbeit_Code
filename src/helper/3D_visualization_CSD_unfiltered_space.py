import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

discrete_values = [0.0090,
                   0.1,
                   0.2,
                   0.3,
                   0.4,
                   0.5]

x = [np.random.choice(discrete_values) for i in range(500)]
y = [np.random.uniform(low=0.2, high=3.1) for i in range(500)]
z = [np.random.randint(1,71, size=500)]

ax.scatter(x, y, z)

ax.set_title('Initialisierungsraum')

ax.set_xlabel('Drahtdurchmesser in Inch')
ax.set_ylabel('Federdurchmesser in Inch')
ax.set_zlabel('Windungsanzahl')

ax.set_xticks(discrete_values)

ax.set_yticks([0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

ax.set_zticks([1, 10, 20, 30, 40, 50, 60, 70])

# plt.savefig('../figures/3D_visualization_CSD_unfiltered_space.png')
plt.show()