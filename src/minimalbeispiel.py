import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from scipy import interpolate
from scipy.integrate import quad
import random

class Minimalbeispiel:

    def __init__(self):

        self._temp_raw = (0, 7.2, 40, 80, 100, 140, 180, 200, 240, 280, 300, 340, 380, 400, 440, 480, 500, 540)
        self._temp_in_kelvin = self.get_temp_in_kelvin(self._temp_raw)

        self._aluminium_temp_in_kelvin = self._temp_in_kelvin
        self._aluminium_thermal_conductivity_raw = (0, 20, 105, 158, 159, 136, 121, 119, 117, 116, 116, 116, 116, 116, 116, 116, 116, 116)
        self._aluminium_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._aluminium_thermal_conductivity_raw)
        self._aluminium_integrand = self.aluminium_integrand

        self._low_carbon_steel_temp_in_kelvin = self._temp_in_kelvin
        self._low_carbon_steel_thermal_conductivity_raw = (0, 1.7, 15, 25.1, 29.5, 34.4, 36.4, 37, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6, 37.6)
        self._low_carbon_steel_thermal_conductivity_in_W_m_K = self.get_thermal_conductivity_in_W_m_K(self._low_carbon_steel_thermal_conductivity_raw)
        self._low_carbon_steel_integrand = self.low_carbon_steel_integrand

        self._aluminium_points = list(zip(self._temp_in_kelvin, self._aluminium_thermal_conductivity_in_W_m_K))
        self._low_carbon_steel_points = list(zip(self._temp_in_kelvin, self._low_carbon_steel_thermal_conductivity_in_W_m_K))

        self._possible_integrands = {0: self._aluminium_integrand,
                                     1: self._low_carbon_steel_integrand
                                     }

    def print_plots(self):
        aluminium_data = np.array(self._aluminium_points)
        low_carbon_steel_data = np.array(self._low_carbon_steel_points)

        tck_aluminium, u_aluminium = interpolate.splprep(aluminium_data.transpose(), s=0)
        tck_low_carbon_steel, u_low_carbon_steel = interpolate.splprep(low_carbon_steel_data.transpose(), s=0)
        unew = np.arange(0, 1.01, 0.01)
        out_aluminium = interpolate.splev(unew, tck_aluminium)
        out_low_carbon_steel = interpolate.splev(unew, tck_low_carbon_steel)

        plt.figure()

        plt.plot(out_aluminium[0], out_aluminium[1], color='orange', label='aluminium')
        plt.plot(aluminium_data[:, 0], aluminium_data[:, 1], 'ob')

        plt.plot(out_low_carbon_steel[0], out_low_carbon_steel[1], color='purple', label='low_carbon_steel')
        plt.plot(low_carbon_steel_data[:, 0], low_carbon_steel_data[:, 1], 'ob')

        plt.xlabel('temperature_in_kelvin')
        plt.ylabel('thermal_conductivity_in_W_m_K')

        plt.legend()

        plt.show()

    def get_init_solution(self) -> (dict, float):

        dim = 1

        # insulators = [random.randint(0, 1) for i in range(dim + 1)]
        insulators = [0, 1]

        # delta_x = np.random.rand(dim + 1)
        # delta_x = delta_x / np.sum(delta_x)
        # assert np.isclose(delta_x.sum(), 1)
        delta_x = [0.5, 0.5]

        # temperatures = [random.random() * 300 for i in range(dim) if (random.random != 0 or random.random != 1)]
        # temperatures.sort()
        # temperatures.insert(0, 0)
        # temperatures.append(300)
        temperatures = [0, 150, 300]

        # print(delta_x)

        solution = {'continuous': [temperatures, delta_x],
                    'ordinal': [],
                    'categorical': insulators
                    }

        return solution, self.get_solution_quality(solution)

    def get_solution_quality(self, solution) -> float:

        total_refr_power = 0

        integrands = solution['categorical']

        delta_x = solution['continuous'][1]
        # delta_x = delta_x / np.sum(delta_x)
        # assert np.isclose(delta_x.sum(), 1)

        t = solution['continuous'][0]

        c = [2.5 for i in range(3)]

        for i in range(2):
            # print(t[i + 1])
            if t[i + 1] < 71:
                c[i] = 4
            if t[i + 1] <= 4.2:
                c[i] = 5

        # print(c)

        # for i in range(1, 2):

        #    total_refr_power = (total_refr_power
        #                        + (c[i - 1] * (300 / t[i] - 1)
        #                           * (1 / delta_x[i] * quad(self._possible_integrands[integrands[i - 1]], t[i], t[i + 1])[0]
        #                              - 1 / delta_x[i - 1] * quad(self._possible_integrands[integrands[i - 1]], t[i - 1], t[i])[0])))

        #    insulator = integrands[i]

        #    I_1 = 1 / delta_x[i] * quad(self._possible_integrands[integrands[i - 1]], t[i], t[i + 1])[0]
        #    I_2 = 1 / delta_x[i -1] * quad(self._possible_integrands[integrands[i - 1]], t[i - 1], t[i])[0]

        #    print(I_1, I_2, insulator)

        print(integrands)
        # print(self._possible_integrands[0])

        # I_1 = 1 / delta_x[0] * quad(self._possible_integrands[integrands[1]], t[1], t[2])[0]
        # I_2 = 1 / delta_x[1] * quad(self._possible_integrands[integrands[0]], t[0], t[1])[0]

        I_1 = quad(self._possible_integrands[integrands[1]], t[1], t[2])[0]
        I_2 = quad(self._possible_integrands[integrands[0]], t[0], t[1])[0]

        total_refr_power += I_1 - I_2

        triangle_area = quad(self.linear_func_integrand, 0, 4)[0]

        print(triangle_area)

        return total_refr_power


    def get_variable_boundaries(self) -> dict:

        variable_boundaries = {'continuous': [[0, 300], [0, 1]],
                               'ordinal': []}

        return variable_boundaries

    @staticmethod
    def get_cubic_spline(x, x_points, y_points):

        tck = interpolate.splrep(x_points, y_points)

        return interpolate.splev(x, tck)

    @staticmethod
    def integrand(x, x_points, y_points):

        return Minimalbeispiel.get_cubic_spline(x, x_points=x_points, y_points=y_points)

    def aluminium_integrand(self, x):

        x_points = self._aluminium_temp_in_kelvin
        y_points = self._aluminium_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def low_carbon_steel_integrand(self, x):

        x_points = self._low_carbon_steel_temp_in_kelvin
        y_points = self._low_carbon_steel_thermal_conductivity_in_W_m_K

        return self.integrand(x, x_points, y_points)

    def linear_func_integrand(self, x):

        x_points = [0, 1, 2, 3]
        y_points = [0, 1, 2, 3]

        return self.integrand(x, x_points, y_points)

    @staticmethod
    def get_temp_in_kelvin(temp_raw):

        temp_in_kelvin = tuple(map(lambda x: x / 1.8, temp_raw))

        return temp_in_kelvin

    @staticmethod
    def get_thermal_conductivity_in_W_m_K(thermal_conductivity_raw: tuple) -> tuple:

        thermal_conductivity_in_W_m_K = tuple(map(lambda x: x / 0.57782, thermal_conductivity_raw))

        return thermal_conductivity_in_W_m_K


# test = Minimalbeispiel()
# test.print_plots()
# print(test.get_init_solution())

