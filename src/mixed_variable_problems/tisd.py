from problem import Problem
import random
import numpy as np
from scipy import interpolate
from scipy.integrate import quad


class TISD(Problem):
    TYPE = 'TISD'
    """
    _t_p = "total refrigeration power required"
    """

    def __init__(self, tisd_name: str, problem_path="../problems/mixed_variable/", load_instance=True) -> None:
        self._problem_type = 'TISD'
        self._tisd_name = tisd_name
        self._problem_path = problem_path

        if load_instance:
            self._instance = open(self._problem_path + tisd_name + '.tisd', "r")
            self._instance_content = self._instance.readlines()
            self._dimension = int(self._instance_content[3][12:])
            self._T_hot = int(self._instance_content[4][8:])
            self._T_cold = int(self._instance_content[5][9:])

            self._t_p = 0
            # TODO: load from file instead of here
            self._possible_insulating_materials = {"T": 1, "N": 2, "F": 3, "E": 4, "S": 5, "A": 6, "L": 7}
            self._temp_raw = (0, 7.2, 40, 80, 100, 140, 180, 200, 240, 280, 300, 340, 380, 400, 440, 480, 500, 540)
            self._temp_in_kelvin = self.get_temp_in_kelvin(self._temp_raw)

            self._aluminium_temp_in_kelvin = self._temp_in_kelvin
            self._aluminium_thermal_conductivity_raw = (0, 20, 105, 158, 159, 136, 121, 119, 117,
                                                        116, 116, 116, 116, 116, 116, 116, 116, 116)
            self._aluminium_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._aluminium_thermal_conductivity_raw)
            self._aluminium_integrand = self.aluminium_integrand

            self._low_carbon_steel_temp_in_kelvin = self._temp_in_kelvin
            self._low_carbon_steel_thermal_conductivity_raw = (0, 1.7, 15, 25.1, 29.5, 34.4, 36.4, 37,
                                                               37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6)
            self._low_carbon_steel_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._low_carbon_steel_thermal_conductivity_raw)
            self._low_carbon_steel_integrand = self.low_carbon_steel_integrand

            self._epoxy_normal_temp_in_kelvin = (0, 20, 77, 195, 297)
            self._epoxy_normal_thermal_conductivity_in_W_m_K = (0, 0.15, 0.2, 0.3, 0.35)
            self._epoxy_normal_integrand = self.epoxy_normal_integrand

            self._epoxy_plane_temp_in_kelvin = (0, 20, 77, 195, 297)
            self._epoxy_plane_thermal_conductivity_in_W_m_K = (0, 0.2, 0.26, 0.44, 0.5)
            self._epoxy_plane_integrand = self.epoxy_plane_integrand

            self._nylon_temp_in_kelvin = self._temp_in_kelvin
            self._nylon_thermal_conductivity_raw = (0, 0.0072, 0.064, 0.125, 0.148, 0.170, 0.185, 0.188, 0.194,
                                                    0.2, 0.202, 0.202, 0.202, 0.202, 0.202, 0.202, 0.202, 0.202)
            self._nylon_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._nylon_thermal_conductivity_raw)
            self._nylon_integrand = self.nylon_integrand

            self._stainless_steel_temp_in_kelvin = self._temp_in_kelvin
            self._stainless_steel_thermal_conductivity_raw = (0, 0.14, 1.26, 3, 3.7, 4.68, 5.49, 5.72, 5.24, 6.53,
                                                              7, 7.22, 7.64, 7.75, 8.02, 8.26, 8.43, 8.67)
            self._stainless_steel_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._stainless_steel_thermal_conductivity_raw)
            self._stainless_steel_integrand = self.stainless_steel_integrand

            self._teflon_temp_in_kelvin = self._temp_in_kelvin
            self._teflon_thermal_conductivity_raw = (0, 0.0266, 0.0855, 0.117, 0.125, 0.135, 0.142, 0.142, 0.143,
                                                     0.146, 0.148, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15)
            self._teflon_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._teflon_thermal_conductivity_raw)
            self._teflon_integrand = self.teflon_integrand

            self._magnitudes_of_categorical_variables = [6]

            self._possible_integrands = {0: self._aluminium_integrand,
                                         1: self._low_carbon_steel_integrand,
                                         2: self._epoxy_normal_integrand,
                                         3: self._epoxy_plane_integrand,
                                         4: self._nylon_integrand,
                                         5: self._stainless_steel_integrand,
                                         6: self._teflon_integrand
                                         }

            self.rng = np.random

            # print(self._aluminium_temp_in_kelvin)
            # print(self._T_hot)
            # print(self._T_cold)
            self._instance.close()

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
        self.rng.seed(0)

    def check_dynamic_change(self, iteration_count: int):

        pass

    def get_magnitudes_of_categorical_variables(self) -> list:

        return self._magnitudes_of_categorical_variables

    def get_variable_boundaries(self) -> dict:

        variable_boundaries = {'continuous': [[0, 300], [0, 1]],
                               'ordinal': []}

        return variable_boundaries

    def init_solution(self) -> (dict, float):
        """
        initialize a new random solution for the TISD problem;

        Returns:
            tuple, float: a possible solution problem instance of a TISD problem,
                          and its solution quality (total refrigeration power P required)
        """
        dim = self._dimension
        condition = False
        self._t_p = 0
        solution = 0

        while not condition:

            # print('in tisd init_solution: ' + str(self._t_p))

            # given n heat intercepts (n+1) insulators are needed
            insulators = [random.randint(0, len(self._possible_insulating_materials) - 1) for i in range(dim+1)]
            # print("in init_solution: " + str(len(insulators)))
            # print([random.randint(1, len(self._possible_insulating_materials)) for i in range(dim+1)])
            # last_insulator = insulators[-1]
            #insulators = insulators.append(self._T_cold)
            # print("in init_solution: " + str(last_insulator))
            # for i in range(len(insulators)-1):

            # for (n+1) intercepts (n+1) thicknesses are needed
            delta_x = np.random.rand(dim+1)
            delta_x = delta_x / np.sum(delta_x)
            assert np.isclose(delta_x.sum(), 1)
            # print(delta_x)
            # print(delta_x.sum())

            temperatures = [random.random()*self._T_hot for i in range(dim) if (random.random != 0 or random.random != 1)]
            temperatures.sort()
            temperatures.insert(0, self._T_cold)
            temperatures.append(self._T_hot)
            # print(temperatures)

            # solution = insulators, delta_x, temperatures, c

            solution = {'continuous': [temperatures, delta_x],
                        'ordinal': [],
                        'categorical': insulators,
                        'abstract': [1]}

            self._t_p = self.get_solution_quality(solution)
            condition = self._t_p[1]
            # print("in Schleife: " + str(solution))
            # print('in tisd init_solution: ' + str(self._t_p))
            # print("in tisd init_solution: " + str(condition))

        # print("außerhalb der Schleife: " + str(solution))

        # return insulators, delta_x, temperatures, c, self.get_solution_quality(solution)
        return solution, self.get_solution_quality(solution)
        # return self.get_solution_quality(solution)

    def get_solution_quality(self, solution: dict) -> (float, bool):
        """
        insulators = "vector_of_insulators"
        delta_x = "vector_of_thicknesses_of_insulators"
        t = "vector_of_temperatures_at_each_intercepts"
        c = "thermodynamic cycle efficiency coefficient at the i-th intercept"

        Args:
            solution:

        Returns: float: total refrigeration power P required

        """
        # print('in tisd: ' + 'test')

        self._t_p = 0.0
        # p_i = 0
        # print("in tisd_get_solution_quality" + " " + str(solution))
        # print("in tisd_get_solution_quality solution[1]" + " " + str(solution[1]))
        # print("in tisd_get_solution_quality 0" + " " + str(solution[0]))
        # print("in tisd_get_solution_quality 1" + " " + str(solution[1]))
        # print("in tisd_get_solution_quality 2" + " " + str(solution[2]))
        # print("in tisd_get_solution_quality 3" + " " + str(solution[3]))
        integrands = solution['categorical']
        delta_x = solution['continuous'][1]
        # print("in tisd get_solution_quality: " + str(delta_x))
        delta_x = delta_x / np.sum(delta_x)
        assert np.isclose(delta_x.sum(), 1)
        # print("in TISD get_solution_quality: " + str(delta_x))
        # print(np.sum(delta_x))
        t = solution['continuous'][0]

        # c = thermodynamic cycle efficiency coefficient at the i-th intercept
        c = [2.5 for i in range(self._dimension)]

        for i in range(len(integrands)):
            if t[i + 1] < 71:
                c[i] = 4
            if t[i + 1] <= 4.2:
                c[i] = 5
        # print(c)

        # c = solution[3]

        # print(self._possible_integrands[integrands[1]])

        # self._t_p = 0

        # print(self._t_p)

        for i in range(1, self._dimension+1):

            # print("in tisd get_solution_quality: " + str(self._possible_integrands[integrands[i-1]]))

            # print(i)
            # print(integrands[1])
            # print(self._t_p)
            # self._t_p = 0

            # integrand = (1/delta_x[i] * quad(self._possible_integrands[integrands[i-1]], t[i], t[i+1])[0]
            #             - 1/delta_x[i-1] * quad(self._possible_integrands[integrands[i-1]], t[i-1], t[i])[0])

            # if integrand <= 0:

            #    return self._t_p, False

            # print(self._T_hot / t[i] - 1)
            # print(1/delta_x[i] * quad(self._possible_integrands[integrands[i]], t[i], t[i+1])[0]
            #      - 1/delta_x[i-1] * quad(self._possible_integrands[integrands[i-1]], t[i-1], t[i])[0])

            # print("in get_solution_quality: " + str(1/delta_x[i] * quad(self._possible_integrands[integrands[i-1]], t[i], t[i+1])[0]))
            # print("in get_solution_quality: " + str(1/delta_x[i-1] * quad(self._possible_integrands[integrands[i-1]], t[i-1], t[i])[0]))

            # print("in get_solution_quality: " + str(integrands[i]))
            # print("in get_solution_quality: " + str(integrands[i-1]))
            # print()

            # print("in get_solution_quality: " + str(self._T_hot/t[i] - 1))
            # print("in get_solution_quality: " + str(c[i-1]))

            p_i = (c[i-1] * (self._T_hot / t[i] - 1)
                   * (1/delta_x[i] * quad(self._possible_integrands[integrands[i]], t[i], t[i+1])[0]
                      - 1/delta_x[i-1] * quad(self._possible_integrands[integrands[i-1]], t[i-1], t[i])[0])
                   )


            print("in get_solution_quality: " + str(p_i))

            if p_i <= 0:

                # print("in get_solution_quality: " + str(p_i))

                return self._t_p, False

            self._t_p = self._t_p + p_i

            # print(c[i-1] * (self._T_hot / t[i] - 1))

            # if self._t_p <= 0:
            #    self._t_p = 0

        # print(self._t_p)

            # print(t[i+1])
            # print(1/delta_x[i] * quad(self._possible_integrands[integrands[i-1]], t[i], t[i+1])[0])
            # print(- 1/delta_x[i-1] * quad(self._possible_integrands[integrands[i-1]], t[i-1], t[i])[0])
        # print(i)
        # print(self._t_p)

        # print("in tisd get_solution_quality: " + str(True))

        return self._t_p, True

    @staticmethod
    def get_cubic_spline(x, x_points, y_points):
        # x_points = [0, 1, 2, 3, 4, 5]
        # y_points = [12, 14, 22, 39, 58, 77]

        tck = interpolate.splrep(x_points, y_points)

        return interpolate.splev(x, tck)

    @staticmethod
    def integrand(x, x_points, y_points):

        return TISD.get_cubic_spline(x, x_points=x_points, y_points=y_points)

    def aluminium_integrand(self, x):

        x_points = self._aluminium_temp_in_kelvin
        y_points = self._aluminium_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def low_carbon_steel_integrand(self, x):

        x_points = self._low_carbon_steel_temp_in_kelvin
        y_points = self._low_carbon_steel_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def epoxy_normal_integrand(self, x):

        x_points = self._epoxy_normal_temp_in_kelvin
        y_points = self._epoxy_normal_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def epoxy_plane_integrand(self, x):

        x_points = self._epoxy_plane_temp_in_kelvin
        y_points = self._epoxy_plane_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def nylon_integrand(self, x):

        x_points = self._nylon_temp_in_kelvin
        y_points = self._nylon_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def stainless_steel_integrand(self, x):

        x_points = self._stainless_steel_temp_in_kelvin
        y_points = self._stainless_steel_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def teflon_integrand(self, x):

        x_points = self._teflon_temp_in_kelvin
        y_points = self._teflon_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    @staticmethod
    def get_temp_in_kelvin(temp_raw: tuple) -> tuple:

        temp_in_kelvin = tuple(map(lambda x: x / 1.8, temp_raw))

        return temp_in_kelvin

    @staticmethod
    def get_thermal_conductivity_in_W_m_K(thermal_conductivity_raw: tuple) -> tuple:

        thermal_conductivity_in_W_m_K = tuple(map(lambda x: x/0.57782, thermal_conductivity_raw))

        return thermal_conductivity_in_W_m_K

    def get_heuristic_component(self, i: int, j: int) -> float:

        pass

    def get_optimal_solution(self):
        
        pass

    def visualize(self, solution=None, interactive=True, filepath=".", ) -> None:

        pass

    @property
    def type(self) -> str:

        return self._problem_type

    @property
    def dimension(self) -> int:

        return self._dimension

    """
    only for tests
    """
    @property
    def temperature(self) -> tuple:

        return self._temp_in_kelvin

    """
    only for tests
    """
    @property
    def conductivity(self) -> tuple:

        return self._aluminium_thermal_conductivity_in_W_m_K

    @property
    def get_aluminium_integrand(self):

        return self._aluminium_integrand

    @dimension.setter
    def dimension(self, value) -> None:

        self._dimension = value

    def get_info(self) -> dict:

        pass

    def __str__(self) -> str:

        pass


# test = TISD("tisd10", "../problems/mixed_variable/")

# print(test.dimension)
# test.get_solution_quality((1,2,3))
# a = test.temperature
# b = test.conductivity
# print(quad(test.get_aluminium_integrand, 0, 1))
# print(range(0, test.dimension))
# test.set_random_seed()
# print(test.init_solution())
