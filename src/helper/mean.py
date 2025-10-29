from math import ceil

read_data_path = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Hom_ACO_R_Very_Simple/artificial_mixed_variable_performance_data/tablet_10_51_mv_adaptive_mixed_variable_10.txt"
save_data_path = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Hom_ACO_R_Very_Simple/artificial_mixed_variable_performance_data/tablet_10_51_mv_adaptive_mixed_variable_10_mean.txt"

with open(read_data_path, "r") as performance_data:

    # reading all the data from file
    lines_in_data = performance_data.read().splitlines()

    restarts = 0
    mean_of_restarts = 1
    sum = 0
    mean = 1
    number_of_entries = 0

    for line in lines_in_data:

        if not line:

            break

        split_line = line.split()

        if split_line[2] != "130000":

            restarts += int(split_line[0])

            # gathering all data in a list
            sum += int(split_line[2])
            number_of_entries += 1

            # print(split_line[2])

    if number_of_entries != 0:

        mean_of_restarts = ceil(restarts/number_of_entries)

        mean = ceil(sum/number_of_entries)

    else:

        mean = 130000

    percentage = number_of_entries/50 * 100

mean_of_runs = open(save_data_path, "w")
mean_of_runs.write(str(mean_of_restarts) + " " + str(mean))
mean_of_runs.write("\n\n")
mean_of_runs.write(str(percentage) + str("%"))
mean_of_runs.close()