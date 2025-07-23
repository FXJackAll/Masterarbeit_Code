import os
import glob

import matplotlib

# matplotlib.use('agg')

import matplotlib.pyplot as plt

'''
makes a box plot of the performance data of the given objective function
:return: none
'''

# print("in performance_box_plot_of_test_function: " + str("klappt"))

objective_function_name = "Paraboloid"

# performance_path_write = data_to_analyze['performance_path_write']
# performance_path_plot_write = data_to_analyze['performance_path_plot_write']

# path where to read the performance data from

read_data_path_1 = "../performance_data/ACO_R_Very_Simple/continuous_performance_data/10_4_10_4/shekel_4_10_fixed_0_continuous_10_4_10_4.txt"
read_data_path_2 = "../performance_data/H_Hom_ACO_R_Very_Simple/continuous_performance_data/10_4_10_4/shekel_4_10_fixed_0_continuous_10_4_10_4.txt"
read_data_path_3 = "../performance_data/H_Het_ACO_R_Very_Simple/continuous_performance_data/10_4_10_4/shekel_4_10_fixed_0_continuous_10_4_10_4.txt"
read_data_path_4 = "../performance_data/H_Het_P_ACO_R_Very_Simple/continuous_performance_data/10_4_10_4/shekel_4_10_fixed_0_continuous_10_4_10_4.txt"
read_data_paths = [read_data_path_1, read_data_path_2, read_data_path_3, read_data_path_4]

# path where to save the box plot
save_pic_path = "../figures/shekel_4_10_fixed_0_comparing_algorithms.png"

algorithm_names = ['VS', 'H_Hom_VS', 'H_Het_VS', 'H_Het_P_VS']

spread = [0 for i in range(len(read_data_paths))]
spread_number = 1

with open(read_data_path_1, "r") as performance_data:

    # reading all the data from file
    lines_in_data = performance_data.read().splitlines()

    temp_spread = [0]

    for line in lines_in_data:

        if not line:

            break

        # print("box_plot_comparing_algorithms line: " + str(line))

        # gathering all data in a list
        temp_spread.append(int(line))

    # removing the first zero from the spread as we started it with zero and then just appended lines
    temp_spread.pop(0)

    spread[0] = temp_spread

for read_data_path in read_data_paths[1:]:

    # works only for Windows
    with open(read_data_path, "r") as performance_data:

        # reading all the data from file
        lines_in_data = performance_data.read().splitlines()

        temp_spread = [0]

        for line in lines_in_data:

            if not line:

                break

            split_line = line.split()

            # gathering all data in a list
            temp_spread.append(int(split_line[2]))

        # removing the first zero from the spread as we started it with zero and then just appended lines
        temp_spread.pop(0)

        spread[spread_number] = temp_spread

        spread_number += 1

# create figure instance
fig = plt.figure(1)

# create axes instance
ax = fig.add_subplot(111)

# creates box plot
# it may look like it's not used but it's necessary
bp = ax.boxplot(spread)

# costom caption
# ax.set_title('Variations of ACO$_\mathbb{R}$-Very-Simple')
ax.set_title('Shekel(4,10) - fixed 0')

# ax.set_subtitle('Paraboloid(6)')

# custom x-axis label
# must be set after box plot is done
ax.set_xticklabels(algorithm_names)

# ax.set_ylim(ymin=0)

plt.show()

# save figure
fig.savefig(save_pic_path)
