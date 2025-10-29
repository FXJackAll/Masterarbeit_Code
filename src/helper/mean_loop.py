import os
from math import ceil

from src.helper.mean import save_data_path

directory = "/home/user/Schreibtisch/Masterarbeit_Code/src/performance_data/H_Hom_ACO_R_Very_Simple/artificial_mixed_variable_performance_data"

# print(directory)

for file in os.listdir(directory):

    file_path = directory + "/" + str(file)

    if "_mean" not in file_path and "adaptive" in file_path:

        file_path_without_txt = os.path.splitext(file_path)[0]

        save_data_path = file_path_without_txt + "_mean.txt"

        # print(file)

        with open(file_path, "r") as performance_data:

            # reading all the data from file
            lines_in_data = performance_data.read().splitlines()

            mean_of_restarts = 0
            sum_of_restarts = 0

            mean_of_iterations = 0
            sum_of_iterations = 0
            number_of_entries = 0

            for line in lines_in_data:

                if not line:

                    break

                split_line = line.split()

                # print(split_line)

                if split_line[1] != "130000":

                    sum_of_restarts += int(split_line[0])

                    # gathering all data in a list
                    sum_of_iterations += int(split_line[1])

                    number_of_entries += 1

                    # print(split_line[2])

            if number_of_entries != 0:

                mean_of_restarts = ceil(sum_of_restarts/number_of_entries)

                mean_of_iterations = ceil(sum_of_iterations/number_of_entries)

            else:

                mean = 130000

            percentage = number_of_entries/50 * 100

        mean_of_runs = open(save_data_path, "w")
        mean_of_runs.write(str(mean_of_restarts) + " " + str(mean_of_iterations))
        mean_of_runs.write("\n\n")
        mean_of_runs.write(str(percentage) + str("%"))
        mean_of_runs.close()