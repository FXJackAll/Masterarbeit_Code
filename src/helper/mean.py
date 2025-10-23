from math import ceil

read_data_path = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Het_P_ACO_R_Very_Simple/continuous_performance_data/10_10/cigar_10_fixed_continuous_10.txt"
save_data_path = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Het_P_ACO_R_Very_Simple/continuous_performance_data/10_10/cigar_10_fixed_continuous_10_mean.txt"

with open(read_data_path, "r") as performance_data:

    # reading all the data from file
    lines_in_data = performance_data.read().splitlines()

    mean = 1
    sum = 0
    number_of_entries = 0

    for line in lines_in_data:

        if not line:

            break

        split_line = line.split()

        # gathering all data in a list
        sum += int(split_line[2])
        number_of_entries += 1

    mean = ceil(sum/number_of_entries)

mean_of_runs = open(save_data_path, "w")
mean_of_runs.write(str(mean))
mean_of_runs.close()