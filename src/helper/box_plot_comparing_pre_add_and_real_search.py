import matplotlib.pyplot as plt

read_data_path_2 = "../performance_data/H_Hom_ACO_R_Very_Simple/mixed_variable_performance_data/csd_3_exact_exact_CSD_fixed_mixed_variable_10_6.txt"
# read_data_path_3 = "../performance_data/H_Het_ACO_R_Very_Simple/mixed_variable_performance_data/csd_3_exact_exact_CSD_fixed_mixed_variable_10_6.txt"
# read_data_path_4 = "../performance_data/H_Het_P_ACO_R_Very_Simple/mixed_variable_performance_data/csd_3_exact_exact_CSD_fixed_mixed_variable_10_6.txt"
# read_data_paths = [read_data_path_1, read_data_path_2, read_data_path_3, read_data_path_4]
# read_data_paths = [read_data_path_2, read_data_path_3, read_data_path_4]

# path where to save the box plot
save_pic_path = "../figures/pre_add_and_real_iterations.png"

algorithm_names = ['pre', 'add', 'real']

spread = [0, 0, 0]
# spread_number = 0

# for read_data_path in read_data_paths:

# works only for Windows
with open(read_data_path_2, "r") as performance_data:

    # reading all the data from file
    lines_in_data = performance_data.read().splitlines()

    temp_spread_pre = [0]
    temp_spread_add = [0]
    temp_spread_real = [0]

    for line in lines_in_data:

        if not line:

            break

        split_line = line.split()

        # print(split_line[1])

        # gathering all data in a list
        temp_spread_pre.append(int(split_line[1]))
        temp_spread_add.append(int(split_line[2]))
        temp_spread_real.append(int(split_line[3] * 13))

    # removing the first zero from the spread as we started it with zero and then just appended lines
    temp_spread_pre.pop(0)
    temp_spread_add.pop(0)
    temp_spread_real.pop(0)

    spread[0] = temp_spread_pre
    spread[1] = temp_spread_add
    spread[2] = temp_spread_real

#    spread_number += 1

# df = pd.DataFrame(spread)

# print(df)

# sns.boxplot(data=df, x="pre-, add- und reale Funktionsauswertungen")

fig = plt.figure(1)

# create axes instance
ax = fig.add_subplot(111)

# spread = [spread, [1, 1]]

# creates box plot
# it may look like it's not used but it's necessary
bp = ax.boxplot(spread)

# costom caption
# ax.set_title('Variations of ACO$_\mathbb{R}$-Very-Simple')
ax.set_title('CSD - exact-exact - fixed 0')

# ax.set_subtitle('Paraboloid(6)')

# custom x-axis label
# must be set after box plot is done
ax.set_xticklabels(algorithm_names)

ax.set_xlabel("pre-, add- und reale Funktionsauswertungen")
# ax.set_ylabel("Anzahl an Funktionsauswertugen")

# ax.set_ylim(ymin=0)

plt.show()

# save figure
# fig.savefig(save_pic_path)
