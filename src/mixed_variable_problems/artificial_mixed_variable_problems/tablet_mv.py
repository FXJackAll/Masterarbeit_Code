import os
from functools import reduce
import json
from src.problem import Problem
from random import choice, uniform
import numpy as np


class Tablet_MV(Problem):
    TYPE = 'artificial_mixed_variable_problem'
    """
    """

    def __init__(self, artificial_mixed_variable_pr_name: str,
                 problem_path="../problems/mixed_variable/artificial_mixed_variable_problems",
                 load_instance=True) -> None:
        self._problem_type = 'artificial_mixed_variable_problem'
        self._artificial_mixed_variable_pr_name = artificial_mixed_variable_pr_name
        self._problem_path = problem_path

        # if load_instance:
        #    self._instance = open(self._problem_path + simple_con_pr_name + '.simple_con_pr', "r")
        #    self._instance_content = self._instance.readlines()
        #    self._dimension = int(self._instance_content[3][12:])
        #    self._left_search_space_boundary = float(self._instance_content[4][29:])
        #    self._right_search_space_boundary = float(self._instance_content[5][30:])

        #    self._magnitudes_of_categorical_variables = [3, 5]

        #    self.rng = np.random

        #    self._instance.close()

        if load_instance:

            with open(os.path.join(self._problem_path, self._artificial_mixed_variable_pr_name + ".json")) as f:
                self._instance = json.load(f)
                self._continuous_dimension = self._instance["continuous_dimension"]
                self._discrete_dimension = self._instance["discrete_dimension"]
                # self._dimension = self._instance["dimension"]
                self._variable_boundaries = self._instance["variable_boundaries"]
                self._left_search_space_boundary = self._variable_boundaries["continuous"][0][0]
                self._right_search_space_boundary = self._variable_boundaries["continuous"][0][1]
                self._discrete_values = self._instance["discrete_values"]
                self._magnitudes_of_categorical_variables = self._instance["magnitudes_of_categorical_variables"]

            self._rng = np.random


    def reset(self) -> None:
        """
        reset the problem to its original state; in this case a random point on the paraboloid
        """

        self.init_solution()

        pass

    def set_dynamic(self, dynamic_intensity_pct=0.2, dynamic_frequency=100, min_iteration_count=2000 - 1) -> None:
        """
        Set the dynamic parameters of the problem, if it's to be a dynamic problem
        """
        pass

    def set_random_seed(self):
        """
        Fixing the seed of the RNG, making the results predictable
        """

        self._rng.seed(0)

        pass

    def check_dynamic_change(self, iteration_count: int):
        """
        Checks if dynamic change should happen in current iteration and triggers it if necessary
        """
        pass

    def get_discrete_values(self) -> list:

        return self._discrete_values

    def get_magnitudes_of_categorical_variables(self) -> list:

        return self._magnitudes_of_categorical_variables

    def get_variable_boundaries(self) -> dict:

        # variable_boundaries = {'continuous': [[self._left_search_space_boundary, self._right_search_space_boundary], [1, 1]],
        #                       'ordinal': []}

        # print(self._variable_boundaries)

        return self._variable_boundaries

    def init_solution(self) -> (dict, float):
        """
        Initialize a new solution for the problem.
        """

        # random point in R^n:
        rand_point = tuple([uniform(self._left_search_space_boundary, self._right_search_space_boundary)
                            for i in range(self._continuous_dimension)])

        value = [choice(self._discrete_values) for i in range(self._discrete_dimension)]

        solution = {'natural_numbers': [],
                    'discrete': [value],
                    'continuous': [rand_point],
                    'ordinal': [],
                    'categorical': []
                    }

        solution["so_far"] = 0

        # print("in Paraboloid_MV init solution: " + str(solution))

        return solution, self.get_solution_quality(solution)

    def get_solution_quality(self, solution: dict) -> (float, bool):
        """
        Get the solution quality for the given possible solution.
        """

        # print("in Paraboloid_MV get_solution_quality: " + str(solution))

        continuous_variables = solution['continuous'][0]
        discrete_variables = solution['discrete'][0]

        # print('in Paraboloid: ' + str(continuous_variables))

        continuous_function_value = 10 ** 4 * continuous_variables[0] ** 2
        discrete_function_value = 0

        for i in range(self._continuous_dimension - 1):

            continuous_function_value += continuous_variables[i + 1] ** 2

        for j in range(self._discrete_dimension):

            discrete_function_value += discrete_variables[j] ** 2

            # print("in ellipsoid_mv get_solution_quality: " + str(discrete_function_value))

        function_value = continuous_function_value + discrete_function_value

        return function_value, True

    def get_heuristic_component(self):
        """
        Get the heuristic component of the problem, in general or for specified indices
        """
        pass

    def get_optimal_solution(self):
        """
        Get the optimal solution (quality) of the problem, if existent
        """
        pass

    def visualize(self):
        """
        Create an interactive view or image from the problem instance
        """
        pass

    def get_info(self) -> dict:
        """
        Get information about the current problem instance as a dict

        Returns:
            dict: Information about the TSP instance
        """
        pass

    @property
    def type(self) -> str:

        return self._problem_type

    @property
    def dimension(self):

        dimension = self._discrete_dimension + self._continuous_dimension

        return dimension

    @dimension.setter
    def dimension(self, value):

        self.dimension = value


# test = Paraboloid('paraboloid10')
# print(test.init_solution())