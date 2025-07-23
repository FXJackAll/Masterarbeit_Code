import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sqrt

# path where to save the box plot
save_pic_path = "../figures/2D_visualization_zakharov_1_-4_10.png"

# Data for plotting
x = np.arange(-5.0, 10.0, 0.01)

f_paraboloid = x**2

f_griewangk = 1 + x**2 / 4000 - cos(x/sqrt(1))

f_zakharov = x**2 + (1/2 * x)**2 + (1/2 * x)**4

fig, ax = plt.subplots()
ax.plot(x, f_zakharov, label='Zakharov 2D')

ax.set_ylim(ymin=0, ymax=800)

ax.set(xlabel='x', ylabel='$f(x)$', title='Zakharov 2D')
ax.grid()

plt.show()

# save figure
fig.savefig(save_pic_path)