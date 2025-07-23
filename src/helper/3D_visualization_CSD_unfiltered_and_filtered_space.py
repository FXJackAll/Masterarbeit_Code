import sys
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from numpy import pi

fig = plt.figure()
ax = plt.axes(projection='3d')

with open("/home/user/Schreibtisch/Masterarbeit/Masterarbeit_XF_OPT_META_Framework/XF-OPT-META/problems/mixed_variable/csd_3.json") as f:
    instance = json.load(f)
    dimension = instance["dimension"]
    additionally_required_iterations = 0
    variable_boundaries = instance["variable_boundaries"]
    N_min = variable_boundaries["natural_numbers"][0][0]  # minimum number of spring coils
    N_max = variable_boundaries["natural_numbers"][0][1]  # maximum number of spring coils
    F_max = instance["constants"]["F_max"]  # maximum working load in pound (lb)
    S = instance["constants"]["S"]  # maximum allowable shear stress in pound per square inch (psi)
    l_max = instance["constants"]["l_max"]  # maximum allowable free length in inch (in)
    d_min = instance["constants"]["d_min"]  # minimum spring wire diameter in inch (in)
    variable_boundaries["continuous"][0][0] = 3 * d_min
    D_min = variable_boundaries["continuous"][0][0]  # minimum outside diameter of the spring in inch (in)
    D_max = variable_boundaries["continuous"][0][1]  # maximum outside diameter of the spring in inch (in)
    F_p = instance["constants"]["F_p"]  # pre-load compression force in pound (lb)
    sigma_pm = instance["constants"]["sigma_pm"]  # allowable maximum deflection under pre-load in inch (in)
    sigma_w = instance["constants"]["sigma_w"]  # deflection from pre-load position to maximum load position in inch (in)
    G = instance["constants"]["G"]  # shear modulus of the material (has no unit)
    d_values = instance["discrete_values"]
    magnitudes_of_categorical_variables = instance["magnitudes_of_categorical_variables"]

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

x = []
y = []
z = []

while len(x) <= 500:

    # number of spring coils N
    N = np.random.randint(1,71) # z-values
    # outside diameter of the spring D
    D = np.random.uniform(low=0.2, high=3.1) # y-values
    # spring wire diameter d
    d = np.random.choice(discrete_values) # x-values

    C_f = (4 * D / d - 1) / (4 * D / d - 4) + (0.615 * d) / D
    K = (G * d ** 4) / (8 * N * D ** 3)
    sigma_p = F_p / K
    l_f = F_max / K + 1.05 * (N + 2) * d

    if (8 * C_f * F_max * D)/(pi * d**3) - S > 0:

        # print('in get_solution_quality: test')

        continue

    if l_f - l_max > 0:

        # print('in get_solution_quality: test2')

        continue

    if D - D_max > 0:

        # print('in get_solution_quality: test3')

        continue

    if 3 - D/d > 0:

        # print('in get_solution_quality: test4')
        # print('in get_solution_quality D: ' + str(D))
        # print('in get_solution_quality d: ' + str(d))
        # print('in get_solution_quality D/d: ' + str(D/d))

        continue

    if sigma_p - sigma_pm > 0:

        # print('in get_solution_quality: test5')

        continue

    if sigma_p + (F_max - F_p)/K + 1.05*(N + 2)*d - l_f > 0:

        # print('in get_solution_quality: test6')

        continue

    if sigma_w - (F_max - F_p)/K > 0:

        # print('in get_solution_quality: test7')

        continue

    if True:

        x.append(d)
        y.append(D)
        z.append(N)

ax.scatter(x, y, z)

ax.set_title('Initialisierungsdata')

ax.set_xlabel('Drahtdurchmesser in Inch')
ax.set_ylabel('Federdurchmesser in Inch')
ax.set_zlabel('Windungsanzahl')

ax.set_xticks(discrete_values)

ax.set_yticks([0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

ax.set_zticks([1, 10, 20, 30, 40, 50, 60, 70])

plt.savefig('../figures/3D_visualization_CSD_unfiltered_and_filtered_space.png')
plt.show()