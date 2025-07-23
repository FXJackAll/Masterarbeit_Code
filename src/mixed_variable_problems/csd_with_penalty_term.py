import sys
import os
from functools import reduce
from operator import mul
import json

from scipy.cluster.hierarchy import weighted

# from collections.abc import set_iterator

from src.problem import Problem
from bisect import bisect
from random import choice, randint, uniform
import numpy as np
from numpy import pi

class CSD_penalty(Problem):
    TYPE = 'CSD'
    """
    _t_p = "total refrigeration power required"
    """

    def __init__(self, csd_name: str, problem_path="../problems/mixed_variable/", load_instance=True) -> None:
        self._problem_type = 'CSD'
        self._csd_name = csd_name
        self._problem_path = problem_path
        self._dimension = 3
        self._epsilon = sys.float_info.epsilon

        if load_instance:

            with open(os.path.join(self._problem_path, self._csd_name + ".json")) as f:
                self._instance = json.load(f)
                self._dimension = self._instance["dimension"]
                self._additionally_required_iterations = 0
                self._variable_boundaries = self._instance["variable_boundaries"]
                self._N_min = self._variable_boundaries["natural_numbers"][0][0] # minimum number of spring coils
                self._N_max = self._variable_boundaries["natural_numbers"][0][1] # maximum number of spring coils
                self._F_max = self._instance["constants"]["F_max"] # maximum working load in pound (lb)
                self._S = self._instance["constants"]["S"] # maximum allowable shear stress in pound per square inch (psi)
                self._l_max = self._instance["constants"]["l_max"] # maximum allowable free length in inch (in)
                self._d_min = self._instance["constants"]["d_min"] # minimum spring wire diameter in inch (in)
                self._variable_boundaries["continuous"][0][0] = 3 * self._d_min
                self._D_min = self._variable_boundaries["continuous"][0][0] # minimum outside diameter of the spring in inch (in)
                self._D_max = self._variable_boundaries["continuous"][0][1] # maximum outside diameter of the spring in inch (in)
                self._F_p = self._instance["constants"]["F_p"] # pre-load compression force in pound (lb)
                self._sigma_pm = self._instance["constants"]["sigma_pm"] # allowable maximum deflection under pre-load in inch (in)
                self._sigma_w = self._instance["constants"]["sigma_w"] # deflection from pre-load position to maximum load position in inch (in)
                self._G = self._instance["constants"]["G"] # shear modulus of the material (has no unit)
                self._d_values = self._instance["discrete_values"]
                self._constraints = [1, 1, 1, 1, 1, 1, 1, 1]
                self._magnitudes_of_categorical_variables = self._instance["magnitudes_of_categorical_variables"]

            self._rng = np.random

    def reset(self) -> None:
        """
        resets the problem to a random initial state
        """

        self.init_solution()

        pass

    def set_dynamic(self, dynamic_intensity_pct=0.2, dynamic_frequency=100, min_iteration_count=2000 - 1) -> None:

        pass

    def set_random_seed(self):
        """
        fixing the seed of the RNG, making the results predictable
        """
        self._rng.seed(0)

    def check_dynamic_change(self, iteration_count: int):

        pass

    def get_discrete_values(self) -> list:

        return self._d_values

    def get_magnitudes_of_categorical_variables(self) -> list:

        return self._magnitudes_of_categorical_variables

    def get_variable_boundaries(self) -> dict:

        # variable_boundaries = {'natural_numbers': [[1, 70]], # TODO: im Paper werden keine Boundaries genannt, obwohl das sehr wichtig ist,
                               # weil das Interval aus dem die ersten k Lösungen gezogen werden, Einfluss hat auf das Verhalten
                               # des Algorithmus
        #                       'discrete': [[self._d_min, 0.5]],
        #                       'continuous': [[self._D_min, self._D_max + self._epsilon]]}

        return self._variable_boundaries

    def init_solution(self) -> (dict, float):
        """
        initialize a new random solution for the TISD problem;

        Returns:
            tuple, float: a possible solution problem instance of a TISD problem,
                          and its solution quality (total refrigeration power P required)
        """
        # dim = self._dimension
        solution = 0

        # print('in csd init_solution: test')

        # solution = insulators, delta_x, temperatures, c

        N = randint(self._N_min, self._N_max + 1) # "self._N_max + 1" because the upper bound of randint is exclusive

        # print('in init_solution: ' + str(N))

        d = choice(self._d_values)
        # print('in csd init_solution: ' + str(d))
        # print('in csd init_solution: ' + str(bool(self._d_min - d > 0)))

        # print('in csd init_solution: test')

        D = uniform(self._D_min, self._D_max)

        solution = {'natural_numbers': [[N]],
                    'discrete': [[d]],
                    'continuous': [[D]],
                    'ordinal': [],
                    'categorical': []}

        solution["so_far"] = 0

        # print("in Schleife: " + str(solution))
        # print('in tisd init_solution: ' + str(self._t_p))
        # print("in tisd init_solution: " + str(condition))

        # print("außerhalb der Schleife: " + str(solution))

        # return insulators, delta_x, temperatures, c, self.get_solution_quality(solution)
        return [solution, self.get_solution_quality(solution)]
        # return self.get_solution_quality(solution)

    def get_solution_quality(self, solution: dict) -> (float, bool):
        """
        Args:
            solution:

        Returns: float: volume of the steel wire required to make the spring coil
        """

        self._constraints = [1, 1, 1, 1, 1, 1, 1, 1]

        # number of spring coils N
        N = solution['natural_numbers'][0][0]
        # outside diameter of the spring D
        D = solution['continuous'][0][0]
        # spring wire diameter d
        d = solution['discrete'][0][0]

        C_f = (4 * D/d - 1)/(4 * D/d - 4) + (0.615 * d)/D
        K = (self._G * d**4)/(8 * N * D**3)
        sigma_p = self._F_p/K
        l_f = self._F_max/K + 1.05 * (N + 2) * d

        g_1 = (8 * C_f * self._F_max * D)/(pi * d**3) - self._S
        g_2 = l_f - self._l_max
        g_3 = self._d_min - d
        g_4 = D - self._D_max
        g_5 = 3 - D/d
        g_6 = sigma_p - self._sigma_pm
        g_7 = sigma_p + (self._F_max - self._F_p)/K + 1.05*(N + 2)*d - l_f
        g_8 = self._sigma_w - (self._F_max - self._F_p)/K

        if g_1 > 0:

            # print("in csd_with_penalty_term g_1: " + str(g_1))

            self._constraints[0] = (self._constraints[0] + 10**(-5) * g_1)**3

        if g_2 > 0:

            # print("in csd_with_penalty_term g_2: " + str(g_2))

            self._constraints[1] = (self._constraints[1] + 1 * g_2)**3

        if g_3 > 0:

            self._constraints[2] = (self._constraints[2] + 10**2 * g_3)**3

        if g_4 > 0:

            self._constraints[3] = (self._constraints[3] + 1 * g_4)**3

        if g_5 > 0:

            self._constraints[4] = (self._constraints[4] + 10**2 * g_5)**3

        if g_6 > 0:

            self._constraints[5] = (self._constraints[5] + 1 * g_6)**3

        if g_7 > 0:

            self._constraints[6] = (self._constraints[6] + 10**2 * g_7)**3

        if g_8 > 0:

            self._constraints[7] = (self._constraints[7] + 10**2 * g_8)**3

        # )print('in get_solution_quality:')

        V_c = 1/4 * pi**2 * D * d**2 * (N + 2)

        # for i in range(len(self._constraints)):

        #    V = V * self._constraints[i]

        V = V_c * reduce(mul, self._constraints)

        # print("in csd_with_penalty_term get_solution_quality: " + str(V))

        return [V, True]

    def get_heuristic_component(self, i: int, j: int) -> float:

        pass

    def get_optimal_solution(self):

        pass

    def visualize(self, solution=None, interactive=True, filepath=".", ) -> None:

        pass

    def req_iterations(self) -> int:

        return self._additionally_required_iterations

    def add_req_iterations(self, additionally_req_iterations) -> None:

        self._additionally_required_iterations += additionally_req_iterations

    @property
    def type(self) -> str:

        return self._problem_type

    @property
    def dimension(self) -> int:

        return self._dimension

    @dimension.setter
    def dimension(self, value) -> None:

        self._dimension = value

    def get_info(self) -> dict:

        pass

    def __str__(self) -> str:

        pass

# test = CSD("csd10", "../problems/mixed_variable/")
# discrete_values = test.get_discrete_values()

# insertion_index = bisect(discrete_values, 0.0333, lo=0, hi=len(discrete_values))
# left_of_discrete_value, right_of_discrete_value = discrete_values[insertion_index], discrete_values[insertion_index - 1]
# discrete_value = left_of_discrete_value if (abs(left_of_discrete_value - 0.0333)
#                                             < abs(right_of_discrete_value - 0.333)) else right_of_discrete_value
# print(discrete_value)
# print(test.dimension)
# test.get_solution_quality((1,2,3))
# for i in range(100):
  #  print(test.init_solution())
# print(test.init_solution())

# print(randint(1,1000))