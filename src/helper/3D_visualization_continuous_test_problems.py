from functools import reduce
from mpl_toolkits import mplot3d
import numpy as np
from mpl_toolkits.mplot3d.axis3d import ZAxis
from numpy import sqrt, pi, cos, exp
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
# from sympy.printing.pretty.pretty_symbology import linewidth

# path where to save the box plot
save_pic_path = "../figures/3D_visualization_hartmann_not_3_2_4_0_1_coolwarm.png"

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def f_b_2_2(x,y):

    function_value = x ** 2 + 2 * y ** 2 - 3 / 10 * cos(3 * pi * x) - 2 / 5 * cos(4 * pi * y) + 7 / 10

    return function_value

def f_branin_rcos_2(x,y):

    function_value = (y - 5 / (4 * pi ** 2) * x ** 2 + 5 / pi * x - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * cos(x) + 10

    return function_value

def f_easom_2(x,y):

    function_value = - cos(x) * cos(y) * exp(- (x - pi)**2 - (y - pi)**2)

    return function_value

def f_goldstein_and_price_2(x,y):

    A = (x + y + 1) ** 2
    B = 19 - 14 * x + 3 * x ** 2
    C = (2 * x - 3 * y) ** 2
    D = 18 - 32 * x + 12 * x ** 2

    function_value = (1 + A * (B - 14 * y + 6 * x * y + 3 * y ** 2)) * (30 + C * (D + 48 * y - 36 * x * y + 27 * y ** 2))

    return function_value

def f_griewangk_2(x,y):

    function_value = (x**2 + y**2)/4000 - cos(x)*cos(y/sqrt(2)) + 1

    return function_value

def f_hartmann_not_3_2_4(x,y):

    continuous_variables = [x,y]

    alpha = [1.0, 1.2, 3.0, 3.2]

    A = [[3.0, 10, 30],
         [0.1, 10, 35],
         [3.0, 10, 30],
        [0.1, 10, 35]]

    P = [[0.3689, 0.1170, 0.2673],
         [0.4699, 0.4387, 0.7470],
         [0.1091, 0.8732, 0.5547],
         [0.0381, 0.5743, 0.8828]]

    # print('in Paraboloid: ' + str(continuous_variables))

    function_value = 0

    for i in range(4):

        term = 0

        for j in range(2):

            term -= A[i][j] * (continuous_variables[j] - P[i][j]) ** 2

        function_value -= alpha[i] * exp(term)

    return function_value

def f_hartmann_not_6_2_4(x,y):

    continuous_variables = [x,y]

    alpha = [1.0, 1.2, 3.0, 3.2]

    A = [[10, 3, 17, 3.5, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]]

    P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
         [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
         [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
         [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]

    # print('in Paraboloid: ' + str(continuous_variables))

    function_value = 0

    for i in range(4):

        term = 0

        for j in range(2):

            term -= A[i][j] * (continuous_variables[j] - P[i][j]) ** 2

        function_value -= alpha[i] * exp(term)

    return function_value

def f_martin_and_gaddy_2(x,y):

    function_value = (x - y) ** 2 + ((x + y - 10) / 3) ** 2

    return function_value

def f_rosenbrock_2(x,y):

    function_value = reduce(lambda a, b: a + b, list(map(lambda x, y: 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2, [x], [y])))

    return function_value

def f_shekel_2_5(x,y):

    continuous_variables = [x,y]

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

    C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
         [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

    function_value = 0
    # sum_in_sum = 0

    for i in range(5):

        sum_in_sum = 0

        for j in range(2):
            sum_in_sum += (continuous_variables[j] - C[j][i]) ** 2

        # print("in 3D_visualization shekel_2_5: " + str(sum_in_sum))
        # print("in 3D_visualization shekel_2_5: " + str(beta))

        # sum_in_sum = sum_in_sum + self._beta[i]

        function_value -= pow((sum_in_sum + beta[i]), -1)

    return function_value

def f_shekel_2_7(x,y):

    continuous_variables = [x,y]

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

    C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
         [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

    function_value = 0
    # sum_in_sum = 0

    for i in range(7):

        sum_in_sum = 0

        for j in range(2):

            sum_in_sum += (continuous_variables[j] - C[j][i]) ** 2

        # print("in 3D_visualization shekel_2_5: " + str(sum_in_sum))
        # print("in 3D_visualization shekel_2_5: " + str(beta))

        # sum_in_sum = sum_in_sum + self._beta[i]

        function_value -= pow((sum_in_sum + beta[i]), -1)

    return function_value

def f_shekel_2_10(x,y):

    continuous_variables = [x,y]

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

    C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
         [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

    function_value = 0
    # sum_in_sum = 0

    for i in range(10):

        sum_in_sum = 0

        for j in range(2):
            sum_in_sum += (continuous_variables[j] - C[j][i]) ** 2

        # print("in 3D_visualization shekel_2_5: " + str(sum_in_sum))
        # print("in 3D_visualization shekel_2_5: " + str(beta))

        # sum_in_sum = sum_in_sum + self._beta[i]

        function_value -= pow((sum_in_sum + beta[i]), -1)

    return function_value

def f_zakharov_2(x,y):

    function_value = x**2 + y**2 + (1/2 * 1 * x + 1/2 * 2 * y)**2 + (1/2 * 1 * x + 1/2 * 2 * y)**4

    return function_value

# Easom
x = np.linspace(-0.0, 1.0, 50, dtype=float)
y = np.linspace(-0.0, 1.0, 50, dtype=float)

X, Y = np.meshgrid(x,y)
Z = f_hartmann_not_3_2_4(X,Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# ax.zaxis.set_major_locator(LinearLocator(10))

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$f(x)$', rotation=5)

ax.set_title('Hartmann(2,4)')

ax.view_init(elev=30, azim=190)

fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.1)

fig.show()

# save figure
fig.savefig(save_pic_path)