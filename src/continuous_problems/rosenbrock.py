import os
from functools import reduce
import json
from src.problem import Problem
from random import uniform
import numpy as np


class Rosenbrock(Problem):
    TYPE = 'continuous_problem'
    """
    """

    def __init__(self, con_pr_name: str,
                 problem_path="../problems/continuous_problems/",
                 load_instance=True) -> None:
        self._con_pr_name = con_pr_name
        self._problem_path = problem_path

        if load_instance:

            with open(os.path.join(self._problem_path, self._con_pr_name + ".json")) as f:
                self._instance = json.load(f)
                self._dimension = self._instance["dimension"]
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

    def init_solution(self) -> (tuple, tuple):
        """
        Initialize a new solution for the problem.
        """

        # random point in R^n:
        rand_point = tuple([uniform(self._left_search_space_boundary, self._right_search_space_boundary)
                            for i in range(self._dimension)])

        solution = {'natural_numbers': [],
                    'discrete': [],
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

        dimension = len(solution)
        # print(solution)
        continuous_variables = solution['continuous'][0]

        function_value = reduce(lambda a, b: a + b, list(map(lambda x, y: 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2,
                                                             continuous_variables[0:dimension - 1], continuous_variables[1:dimension])))

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

        return 'continuous_problem'

    @property
    def dimension(self):

        return self._dimension

    @dimension.setter
    def dimension(self, value):

        self.dimension = value
