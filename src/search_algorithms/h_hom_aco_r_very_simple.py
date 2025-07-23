import random
from functools import reduce
from bisect import bisect
from random import uniform
# from unittest.mock import right

from src.search_algorithm import Search_Algorithm


class H_Hom_ACO_R_Very_Simple(Search_Algorithm):

    def construct_solution(self, search_parameters: dict) -> dict:
        problem = search_parameters['problem']
        search_space_dimension = problem.dimension

        sce_index = search_parameters['sce_index']

        solution_creating_behavior = search_parameters['solution_creating_behavior']

        linear_div_inflation = 1

        raw_populations = search_parameters['populations']

        w_personal_previous = search_parameters['w_personal_previous']
        w_personal_best = search_parameters['w_personal_best']
        w_parent_best = search_parameters['w_parent_best']

        # print("in aco_r_very_simple: " + str(len(raw_populations[2]['continuous'])))

        natural_numbers_populations = [raw_populations[g]['natural_numbers'] for g in range(len(raw_populations))]
        continuous_populations = [raw_populations[h]['continuous'] for h in range(len(raw_populations))]
        discrete_populations = [raw_populations[i]['discrete'] for i in range(len(raw_populations))]
        ordinal_populations = [raw_populations[j]['ordinal'] for j in range(len(raw_populations))]
        categorical_populations = [raw_populations[k]['categorical'] for k in range(len(raw_populations))]

        # print('in aco_r_very_simple: ' + str(continuous_populations))
        # print("in aco_r_very_simple: " + str(continuous_populations[0]))

        best_solution = search_parameters['best_solution']

        discrete_values = problem.get_discrete_values()

        variable_boundaries = problem.get_variable_boundaries()

        natural_numbers_variable_boundaries = variable_boundaries['natural_numbers']
        discrete_variable_boundaries = variable_boundaries['discrete']
        continuous_variable_boundaries = variable_boundaries['continuous']

        # print("in aco_very_simple: " + str(continuous_variable_boundaries))

        magnitude_of_categorical_variables = problem.get_magnitudes_of_categorical_variables()

        # linear_div_inflation = 1

        new_natural_numbers_solution_coordinates = [[] for i in range(len(natural_numbers_populations[0]))]
        new_continuous_coordinates = [[] for i in range(len(discrete_populations[0]))]
        new_discrete_solution_coordinates = [[] for i in range(len(discrete_populations[0]))]
        new_continuous_solution_coordinates = [[] for i in range(len(continuous_populations[0]))]
        new_categorical_choices = [[] for i in range(len(categorical_populations[0]))]

        # print('in aco_r_very_simple:' + str(new_continuous_solution_coordinates))

        natural_numbers_matrix = [[] for dim in range(len(natural_numbers_populations[0]))]
        discrete_matrix = [[] for dim in range(len(discrete_populations[0]))]
        continuous_matrix = [[] for dim in range(len(continuous_populations[0]))]

        # print('in aco_r_very_simple:' + str(continuous_populations[0]))

        for variable in range(len(natural_numbers_populations[0])):

            for solution in range(len(natural_numbers_populations)):

                natural_numbers_matrix[variable].append(natural_numbers_populations[solution][variable])

            number_of_natural_numbers_rows = len(natural_numbers_matrix[variable])

            natural_numbers_means_row_vector = best_solution[0]['natural_numbers'][variable]

            linear_div_inflated_natural_numbers = [0 for dim in range(len(natural_numbers_matrix[variable][0]))]

            # print('in aco_r_very_simple: ' + str(linear_div_inflated))

            # loop repeats for every dimension of the objective function/dimension of the first solution in solution archive
            for dim in range(len(linear_div_inflated_natural_numbers)):
                # in paper row was named "s_e" as other solutions in solution archive
                # linear deviation of solutions in archive from chosen solution
                # takes out of the solution archive/matrix with solutions the i-th coordinate out of the j-th row
                linear_deviation_in_i_th_dimension = [abs(row[dim] - natural_numbers_means_row_vector[dim]) for row in
                                                      natural_numbers_matrix[variable]]

                sum_of_linear_deviations = reduce(lambda a, b: a + b, linear_deviation_in_i_th_dimension)

                # inflated linear deviation
                linear_div_inflated_natural_numbers[dim] = linear_div_inflation * (
                            1 / (number_of_natural_numbers_rows - 1)) * sum_of_linear_deviations

            lin_deviation_vector = linear_div_inflated_natural_numbers

            left_boundary = [natural_numbers_variable_boundaries[variable][0] for i in range(len(linear_div_inflated_natural_numbers))]
            right_boundary = [natural_numbers_variable_boundaries[variable][1] for i in range(len(linear_div_inflated_natural_numbers))]

            # print("in aco_r_very_simple: " + str(left_boundary))

            new_natural_numbers_solution_coordinates[variable] = [
                round(uniform(max(natural_numbers_means_row_vector[i] - lin_deviation_vector[i], left_boundary[i]),
                              min(natural_numbers_means_row_vector[i] + lin_deviation_vector[i], right_boundary[i])))
                for i in range(len(linear_div_inflated_natural_numbers))]



        for variable in range(len(discrete_populations[0])):

            for solution in range(len(discrete_populations)):

                discrete_matrix[variable].append(discrete_populations[solution][variable])

            number_of_discrete_rows = len(discrete_matrix[variable])

            discrete_means_row_vector = best_solution[0]['discrete'][variable]

            linear_div_inflated_discrete = [0 for dim in range(len(discrete_matrix[variable][0]))]

            for dim in range(len(linear_div_inflated_discrete)):

                linear_deviation_in_i_th_dimension = [abs(row[dim] - discrete_means_row_vector[dim]) for row in
                                                      discrete_matrix[variable]]

                sum_of_linear_deviations = reduce(lambda a, b: a + b, linear_deviation_in_i_th_dimension)

                linear_div_inflated_discrete[dim] = linear_div_inflation * (
                    1 / (number_of_discrete_rows - 1)) * sum_of_linear_deviations

            lin_deviation_vector = linear_div_inflated_discrete

            left_boundary = [discrete_variable_boundaries[variable][0] for i in range(len(linear_div_inflated_discrete))]
            right_boundary = [discrete_variable_boundaries[variable][1] for i in range(len(linear_div_inflated_discrete))]

            new_continuous_coordinates[variable] = [uniform(max(discrete_means_row_vector[i] - lin_deviation_vector[i], left_boundary[i]),
                                                            min(discrete_means_row_vector[i] + lin_deviation_vector[i], right_boundary[i]))
                                                    for i in range(len(linear_div_inflated_discrete))]

            for i in range(len(new_continuous_coordinates[variable])):

                insertion_index = bisect(discrete_values, new_continuous_coordinates[variable][i], lo=1, hi=len(discrete_values)-1)
                # print("in ACO_R_Very_Simple: " + str(insertion_index))
                left_of_discrete_value, right_of_discrete_value = discrete_values[insertion_index], discrete_values[insertion_index - 1]
                discrete_value = left_of_discrete_value if (abs(left_of_discrete_value - new_continuous_coordinates[variable][i])
                                                            < abs(right_of_discrete_value - new_continuous_coordinates[variable][i])) else right_of_discrete_value

                new_discrete_solution_coordinates[variable].append(discrete_value)



        for variable in range(len(continuous_populations[0])):

            for solution in range(len(continuous_populations)):

                # print('in aco_r_very_simple: ' + str(continuous_populations[solution]))

                continuous_matrix[variable].append(continuous_populations[solution][variable])

            number_of_continuous_rows = len(continuous_matrix[variable])

            # print('in aco_r_very_simple: ' + str(best_solution))

            continuous_means_row_vector = best_solution[0]['continuous'][variable]

            # print('in aco_r_very_simple: ' + str(matrix))

            # solution archive interpreted as matrix
            # matrix = continuous_populations
            # number of solutions in solution archive
            # number_of_rows = len(matrix)

            # the chosen solution vector, named in paper as "s_L"; actually small L but that looks like the number "one"
            # means_row_vector = best_solution[0]['continuous'][0]

            # print(means_row_vector)

            # linear_div_inflation = linear_deviation_parameters['linear_div_inflation']

            linear_div_inflated = [0 for dim in range(len(continuous_matrix[variable][0]))]

            # print('in aco_r_very_simple: ' + str(linear_div_inflated))

            # loop repeats for every dimension of the objective function/dimension of the first solution in solution archive
            for dim in range(len(linear_div_inflated)):
                # in paper row was named "s_e" as other solutions in solution archive
                # linear deviation of solutions in archive from chosen solution
                # takes out of the solution archive/matrix with solutions the i-th coordinate out of the j-th row
                linear_deviation_in_i_th_dimension = [abs(row[dim] - continuous_means_row_vector[dim]) for row in continuous_matrix[variable]]

                sum_of_linear_deviations = reduce(lambda a, b: a + b, linear_deviation_in_i_th_dimension)

                # inflated linear deviation
                linear_div_inflated[dim] = linear_div_inflation * (1 / (number_of_continuous_rows - 1)) * sum_of_linear_deviations

            lin_deviation_vector = linear_div_inflated

            left_boundary = [continuous_variable_boundaries[variable][0] for i in range(len(linear_div_inflated))]
            right_boundary = [continuous_variable_boundaries[variable][1] for i in range(len(linear_div_inflated))]

            # print("in aco_r_very_simple: " + str(left_boundary))

            new_continuous_solution_coordinates[variable] = [uniform(max(continuous_means_row_vector[i] - lin_deviation_vector[i], left_boundary[i]),
                                                                     min(continuous_means_row_vector[i] + lin_deviation_vector[i], right_boundary[i]))
                                                             for i in range(len(linear_div_inflated))]


        # print('in aco_r_very_simple:' + str(categorical_populations))

        # print("in aco_r_very_simple: " + str(search_space_dimension))

        for magnitude in range(len(magnitude_of_categorical_variables)):

            for variable in range(len(categorical_populations[0])):

                # print("in aco_very_simple len: " + str(len(categorical_populations[0])))

                # print("in aco_very_simple categorical_populations[0]: " + str(categorical_populations[0]))

                # print("in aco_very_simple variable: " + str(variable))

                # print("in aco_very_simple magnitude_of_cat: " + str(magnitude_of_categorical_variables[magnitude]))

                new_categorical_choices[variable] = random.randint(0, magnitude_of_categorical_variables[magnitude])

        new_solution = {'natural_numbers': new_natural_numbers_solution_coordinates,
                        'discrete': new_discrete_solution_coordinates,
                        'continuous': new_continuous_solution_coordinates,
                        'ordinal': [],
                        'categorical': new_categorical_choices
                        }

        # print("in aco_r_very_simple: " + str(new_categorical_choices))

        # print(problem.init_solution()[1])

        return new_solution

# print(max(1, True))