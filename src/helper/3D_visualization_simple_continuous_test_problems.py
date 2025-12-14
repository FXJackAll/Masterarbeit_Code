import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# path where to save the box plot
save_pic_path = "../figures/3D_visualization_paraboloid_2_-3_7_coolwarm.png"

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def f_cigar_2(y,x):

    function_value = x ** 2 + 10 ** 4 * y ** 2

    return function_value

def f_ellipsoid_2(x,y):

    continuous_variables = [x,y]

    function_value = 0

    for i in range(2):

        function_value += (100 ** ((i - 1) / (2 - 1)) * continuous_variables[i]) ** 2

    return function_value

def f_paraboloid_2(x,y):

    function_value = x**2 + y**2

    return function_value

# Easom
x = np.linspace(-3.0, 7.0, 50, dtype=float)
y = np.linspace(-3.0, 7.0, 50, dtype=float)

X, Y = np.meshgrid(x,y)
Z = f_paraboloid_2(X,Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$f(x)$', rotation=5)

ax.set_title('Praboloid')

ax.view_init(elev=30, azim=235)

fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.1)

fig.show()

# save figure
fig.savefig(save_pic_path)