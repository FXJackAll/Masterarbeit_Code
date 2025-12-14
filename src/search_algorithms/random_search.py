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

        possible_solutions = (random_solution[0],
                              populations[0],
                              populations[1],
                              populations[2])

        new_solution = random.choices(possible_solutions, random_distribution)

        return new_solution[0]
