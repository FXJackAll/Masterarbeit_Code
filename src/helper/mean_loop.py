import os
from math import ceil

from src.helper.mean import save_data_path

directory = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Het_P_ACO_R_Very_Simple/artificial_mixed_variable_performance_data"

# print(directory)

for file in os.listdir(directory):

    file_path = directory + "/" + str(file)

    if "_mean" not in file_path:

        file_path_without_txt = os.path.splitext(file_path)[0]

        save_data_path = file_path_without_txt + "_mean.txt"

        # print(file)

        with open(file_path, "r") as performance_data:

            # reading all the data from file
            lines_in_data = performance_data.read().splitlines()

            mean = 1
            sum = 0
            number_of_entries = 0

            for line in lines_in_data:

                if not line:

                    break

                split_line = line.split()

                # print(split_line)

                if split_line[2] != "130000":

                    # gathering all data in a list
                    sum += int(split_line[2])
                    number_of_entries += 1

                    # print(split_line[2])

            if number_of_entries != 0:

                mean = ceil(sum/number_of_entries)

            else:

                mean = 130000

            percentage = number_of_entries/50 * 100

        mean_of_runs = open(save_data_path, "w")
        mean_of_runs.write(str(mean))
        mean_of_runs.write("\n\n")
        mean_of_runs.write(str(percentage) + str("%"))
        mean_of_runs.close()