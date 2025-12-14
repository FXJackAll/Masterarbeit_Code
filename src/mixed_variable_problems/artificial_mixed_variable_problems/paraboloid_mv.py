import os
from functools import reduce
import json
from src.problem import Problem
from random import choice, uniform
import numpy as np


class Paraboloid_MV(Problem):
    TYPE = 'artificial_mixed_variable_problem'
    """
    """

    def __init__(self, artificial_mixed_variable_pr_name: str,
                 problem_path="../problems/mixed_variable_problems/artificial_mixed_variable_problems",
                 load_instance=True) -> None:
        self._problem_type = 'artificial_mixed_variable_problem'
        self._artificial_mixed_variable_pr_name = artificial_mixed_variable_pr_name
        self._problem_path = problem_path

        if load_instance:

            with open(os.path.join(self._problem_path, self._artificial_mixed_variable_pr_name + ".json")) as f:
                self._instance = json.load(f)
                self._continuous_dimension = self._instance["continuous_dimension"]
                self._discrete_dimension = self._instance["discrete_dimension"]
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

        return solution, self.get_solution_quality(solution)

    def get_solution_quality(self, solution: dict) -> (float, bool):
        """
        Get the solution quality for the given possible solution.
        """

        continuous_variables = solution['continuous'][0]
        discrete_variables = solution['discrete'][0]

        continuous_function_value = reduce(lambda a, b: a + b, list((map(lambda x: x ** 2, continuous_variables))))
        discrete_function_value = reduce(lambda a, b: a + b, list(map(lambda x: x ** 2, discrete_variables)))

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

    def req_iterations(self) -> int:

        return 0

    def add_req_iterations(self, additionally_required_iterations) -> None:

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