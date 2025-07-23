import os
import glob

import matplotlib

# matplotlib.use('agg')

import matplotlib.pyplot as plt

'''
makes a box plot of the performance data of the given objective function
:return: none
'''

print("in performance_box_plot_of_test_function: " + str("klappt"))

objective_function_name = "Paraboloid"

# performance_path_write = data_to_analyze['performance_path_write']
# performance_path_plot_write = data_to_analyze['performance_path_plot_write']

# path where to read the performance data from
read_data_path = "../performance_data/H_Hom_ACO_R_Very_Simple/continuous_performance_data/10_10/paraboloid_10_fixed_continuous_10.txt"

# path where to save the box plot
save_pic_path = "../figures/paraboloid_10.png"

# works only for Windows
with open(read_data_path, "r") as performance_data:

    # reading all the data from file
    lines_in_data = performance_data.read().splitlines()

    spread = [0]

    for line in lines_in_data:

        if not line:

            break

        split_line = line.split()

        # gathering all data in a list
        spread.append(int(split_line[2]))

    # removing the first zero from the spread as we started it with zero and then just appended lines
    spread.pop(0)

    # create figure instance
    fig = plt.figure(1)

    # create axes instance
    ax = fig.add_subplot(111)

    # creates box plot
    # it may look like it's not used but it's necessary
    bp = ax.boxplot(spread)

    # custom x-axis label
    # must be set after box plot is done
    ax.set_xticklabels([objective_function_name])

    plt.show()

    # save figure
    # fig.savefig(save_pic_path)
