from src.search_algorithm import Search_Algorithm
import random


class Random_Search(Search_Algorithm):

    def construct_solution(self, search_parameters: dict):
        problem = search_parameters['problem']
        populations = search_parameters['populations']
        w_personal_previous = search_parameters['w_personal_previous']
        w_personal_best = search_parameters['w_personal_best']
        w_parent_best = search_parameters['w_parent_best']

        random_distribution = [1 - (w_personal_previous + w_personal_best + w_parent_best),
                               w_personal_previous, w_personal_best, w_parent_best]

        random_solution = problem.init_solution()

        # print(populations[0])

        possible_solutions = (random_solution[0],
                              populations[0],
                              populations[1],
                              populations[2])

        # print(random_solution)

        new_solution = random.choices(possible_solutions, random_distribution)

        # print("in random_search: " + str(new_solution))

        # print(new_solution)

        # print(problem.init_solution()[1])

        # print("in random_search: ")
        # print(new_solution[0])

        # test = [new_solution[0][0], new_solution[0][1], new_solution[0][2], new_solution[0][3]]

        return new_solution[0]
