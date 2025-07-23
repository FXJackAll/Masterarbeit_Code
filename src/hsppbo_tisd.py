import numpy as np
import random
from os import urandom
from mpire import WorkerPool
from math import pow
from sce_tree_tisd import SCETree_TISD
from problem import Problem
from search_algorithm import Search_Algorithm
from logger import Logger
from functools import partial
from config import params


class HSPPBO_TISD:

    #def __init__(self, problem: Problem, search_algorithms: Search_Algorithm, logger: Logger,
    #             pers_best=0.075, pers_prev=0.075, parent_best=0.01,
    #             alpha=1, beta=9, detection_threshold=0.25, reaction_type='partly', detection_pause=5,
    #             solution_creating_behaviour='weak-heterogeneous',
    #             max_iteration_count=2600) -> None:

    def __init__(self, HSPPBO_parameters) -> None:
        """
        Initialize the hsppbo algorithm with the given parameters.

        Args:
            problem (Problem): The problem instance that the algo is searching solutions for
            logger (Logger): The logger used during the algorithm run
            pers_best (float, optional): Influence of the SCE\'s best personal solution ever obtained. Defaults to 0.075.
            pers_prev (float, optional): Influence of the SCE\'s previous personal solution. Defaults to 0.075.
            parent_best (float, optional): Influence of the SCE\'s parent personal best solution ever obtained. Defaults to 0.01.
            alpha (int, optional): Influence of the probabilistic/non-heuristic component. Defaults to 1.
            beta (int, optional): Influence of the heuristic component. Defaults to 5.
            detection_threshold (float, optional): Threshold (swaps per SCE and iteration) for detecting a change. Defaults to 0.25.
            detection_pause (int, optional): Min number of iterations between change handling procedures (variable L in literaute). Defaults to 5.
            reaction_type (str, optional): Type of reaction algorithm used to handle a change in the problem. Defaults to 'partial'.
            max_iteration_count (int, optional): Maximum number of iterations per algorithm run. Defaults to 2600.
        """
        self._problem = HSPPBO_parameters['problem']
        self._logger = HSPPBO_parameters['logger']

        self._search_algorithm = HSPPBO_parameters['search_algorithm']
        self._restart_behavior = HSPPBO_parameters['restart_behaviour']
        # self._temp_solution = [[0, 10**10],0]
        self._period = HSPPBO_parameters['period']
        self._adaptive_threshold = HSPPBO_parameters['adaptive_threshold']
        self._required_restarts = 0
        self._problem_optimum = HSPPBO_parameters['problem_optimum']
        self._accuracy = HSPPBO_parameters['accuracy']
        self._absolute_accuracy = HSPPBO_parameters['absolute_acc']
        self._relative_accuracy = HSPPBO_parameters['relative_acc']

        self._sce_count = HSPPBO_parameters['SCE_parameters']['sce_count']
        self._sce_child_count = HSPPBO_parameters['SCE_parameters']['sce_child_count']
        self._tree = SCETree_TISD(self._problem.init_solution, self._sce_count, self._sce_child_count)
        self._w_pers_best = HSPPBO_parameters['SCE_parameters']['w_personal_best']
        self._w_pers_prev = HSPPBO_parameters['SCE_parameters']['w_personal_previous']
        self._w_parent_best = HSPPBO_parameters['SCE_parameters']['w_parent_best']

        self._w_rand = 1 / (self._problem.dimension - 1)
        self._accepted_solutions = HSPPBO_parameters['accepted_solutions']
        self._solution_creating_behaviour = HSPPBO_parameters['solution_creating_behaviour']
        self._linear_div_inflations = HSPPBO_parameters['linear_div_inflations']
        self._h_probabilities = HSPPBO_parameters['h_probabilities']

        self._alpha = HSPPBO_parameters['TSP_parameters']['alpha']
        self._beta = HSPPBO_parameters['TSP_parameters']['beta']

        self._detection_threshold = HSPPBO_parameters['dynamic_detection_threshold']
        self._detection_pause = HSPPBO_parameters['detection_pause']
        self._reaction_type = HSPPBO_parameters['reaction_type']

        self._max_iteration_count = HSPPBO_parameters['max_iteration_count']

        self._solution_quality_intervals = []

        self._fixed_rng = False

    def set_random_seed(self):
        """
        Fixing the seed of the RNG, making the results predictable
        """
        random.seed(0)
        self._problem.set_random_seed()
        self._fixed_rng = True

    def execute_wrapper(self, *args) -> int:
        """
        Args:
            *args (optional):

        Returns:
            int: Average of best solution (tuple of path and length) found during the runtime of the algorithm.
        """

        # set the params for the algorithm run
        for k, v in enumerate(args[0]):
            self.__dict__[params['opt']['hsppbo'][k][0]] = v

        # for multiple runs, the tree and solution list need to be reset
        self._tree.reset()
        self._problem.reset()
        self._solution_quality_intervals.clear()

        self.execute()

        # calc average of (best) solutions right before dynamic is triggered
        avg_solution_quality = np.mean(self._solution_quality_intervals)
        return avg_solution_quality

    def reset_tree(self):

        self._tree.reset()

    def reset_problem(self):

        self._required_restarts = 0

    def execute(self, verbose=False) -> (dict, int):
        """
        Execute the HSPPBO algorithm

        Args:
            verbose (bool, optional): Show verbose outputs during and after the algorithm runs. Defaults to False.

        Returns:
            tuple[list, int]: Best solution (tuple of path and length) found during the runtime of the algorithm.
        """

        self._tree.init_tree()
        detection_pause_count = self._detection_pause

        # self._required_restarts = 0

        for i in range(0, self._max_iteration_count):

            temp_solution = self._tree.get_best_solution(self._tree.tree.root)[1]

            # print('in HSPPPBO_TISD execute: ' + 'test')

            # check for dynamic change within the problem instance
            # if dynamic is triggered (true), recalculate the solution quality for each node
            if self._problem.check_dynamic_change(i):
                self._solution_quality_intervals.append(self._tree.get_best_solution(self._tree.tree.root)[1])

                for sce in range(0, self._sce_count):
                    pp_quality = self._problem.get_solution_quality(self._tree.get_solution(sce)[0])
                    pb_quality = self._problem.get_solution_quality(self._tree.get_best_solution(sce)[0])
                    print("in HSPPBO_TISD execute pb_quality: " + str(pb_quality))
                    self._tree.set_node_quality(sce, pp_quality, pb_quality)

            # count up the change pause counter up to the constant
            if detection_pause_count < self._detection_pause:
                detection_pause_count += 1

            if (self._problem.dimension > 100):
                with WorkerPool(n_jobs=self._sce_count) as pool:
                    # call the same function with different data in parallel, pass the current iteration as partial function
                    for sce, solution in pool.map_unordered(partial(self.find_next_solution, iteration=i), range(0, self._sce_count)):
                        solution_quality = self._problem.get_solution_quality(solution)[0]
                        self._tree.update_node(sce, solution, solution_quality)
            else:
                # NON-MULTITHREADED VERSION
                for sce in range(0, self._sce_count):
                    solution = self.find_next_solution(sce, i)[1]
                    # print("in HSPPBO_TISD execute: " + str(solution))
                    # print("in HSPPBO_TISD execute: " + str(solution))
                    solution_quality = self._problem.get_solution_quality(solution)[0]
                    # print("in HSPPBO_TIDS execute solution_quality : " + str(solution_quality))
                    self._tree.update_node(sce, solution, solution_quality)

            swap_count = 0  # number of swaps performed in the SCE tree
            for sce in self._tree.all_nodes_top_down:
                # get the pers best solution of the sce and its parent
                sce_solution_quality = self._tree.get_best_solution(sce)[1]
                # print("in HSPPBO_TISD execute: " + str(sce_solution_quality))
                # get the solution quality of the sce and its parent
                parent_sce = self._tree.get_parent(sce)
                parent_solution_quality = self._tree.get_best_solution(parent_sce)[1]
                # print("in HSPPBO_TISD execute: " + str(parent_solution_quality))
                # print("hsppbo_tisd execute: " + str(sce_solution_quality))
                # swap parent and sce position, if sce solution is better
                if sce_solution_quality < parent_solution_quality:
                    self._tree.swap_nodes(sce, parent_sce)
                    swap_count += 1

            # if the threshold for detecting a dynamic change is met,
            # trigger the change handling procedure of the tree and reset the change pause counter
            if swap_count > (self._detection_threshold * self._sce_count):
                if detection_pause_count == self._detection_pause:
                    detection_pause_count = 0
                    self._tree.change_handling(self._reaction_type)

            if i % 10 == 0 and verbose:
                print("Iteration: ", i, "\n")
                self._tree.tree.show(data_property="pb_quality")

            best_solution = self._tree.get_best_solution(self._tree.tree.root)[0]
            best_solution["required_restarts"] = self._required_restarts
            # print("in hsspbo_tisd execute: " + str(best_solution))
            best_solution_quality = self._tree.get_best_solution(self._tree.tree.root)[1]
            # print("in hsppbo_tisd execute: " + str(best_solution_quality))
            self._logger.log_iteration(i, self._sce_count * (i + 1), swap_count, detection_pause_count == 0, best_solution_quality, best_solution)

            match self._accuracy:

                case 'absolute':

                    # print("in HSPPBO_TISD execute: " + str(best_solution_quality))

                    if np.abs(best_solution_quality - self._problem_optimum) < self._absolute_accuracy:

                        # print("in HSPPBO_TISD execute: " + str(best_solution_quality))

                        best_solution = self._tree.get_best_solution(self._tree.tree.root)
                        # append last solution quality - without dynamic, this is the only list entry
                        self._solution_quality_intervals.append(best_solution[1])
                        best_solution[0]["required_restarts"] = self._required_restarts

                        if verbose:
                            self._tree.tree.show(data_property="pb_quality")
                            self._problem.visualize(solution=best_solution[0])

                        # if "so_far" not in best_solution:

                           # best_solution["so_far"] = self._problem.req_iterations()

                        return [best_solution, i]

                case 'relative':

                    # print('in HSPPBO tisd execute ' + 'test')

                    if (np.abs(best_solution_quality - self._problem_optimum)
                            < (np.abs(self._relative_accuracy * best_solution_quality) + self._relative_accuracy)):

                        # print("in HSPPBO_TISD execute: " + str(np.abs(self._relative_accuracy * solution_quality) + self._relative_accuracy))
                        # print("in HSPPBO_TISD execute: " + str(self._relative_accuracy))

                        # print(np.abs(solution_quality - self._problem_optimum))

                        best_solution = self._tree.get_best_solution(self._tree.tree.root)
                        # append last solution quality - without dynamic, this is the only list entry
                        self._solution_quality_intervals.append(best_solution[1])
                        best_solution[0]["required_restarts"] = self._required_restarts

                        if verbose:
                            self._tree.tree.show(data_property="pb_quality")
                            self._problem.visualize(solution=best_solution[0])

                        # if "so_far" not in best_solution[0][0]:

                          #  best_solution[0][0]["so_far"] = self._problem.req_iterations()

                        return [best_solution, i]

                case _:

                    pass

            if (self._restart_behavior == 'adaptive' and i%self._period == 0):

                # print("in HSPPBO_TISD execute: " + str(temp_solution))
                # print("in HSPPBO_TISD execute: " + str(best_solution) + " " + str(best_solution_quality))

                if abs(best_solution_quality - temp_solution) < self._adaptive_threshold:

                    self._tree.reset()
                    self._tree.init_tree()
                    # print("in HSPPBO_TISD execute: " + str(self._required_restarts))
                    self._required_restarts += 1

                    # print("in HSPPBO_TISD execute: klappt")

        best_solution = self._tree.get_best_solution(self._tree.tree.root)
        best_solution[0]["required_restarts"] = self._required_restarts
        # print("in HSPPBO_TISD execute: " + str(best_solution[0]))
        # append last solution quality - without dynamic, this is the only list entry
        self._solution_quality_intervals.append(best_solution[1])

        if verbose:
            self._tree.tree.show(data_property="pb_quality")
            self._problem.visualize(solution=best_solution[0])

        # print('test')

        # if "so_far" not in best_solution[0][0]:

          #  best_solution[0][0]["so_far"] = self._problem.req_iterations()

        return [best_solution, self._max_iteration_count]

    def find_next_solution(self, sce_index: int, iteration: int) -> tuple[int, tuple[float]]:
        """
        Constructs the solution path (tuple of node indices) according to the algorithm of the HSPPBO paper

        Args:
            sce_index (int): Index of the SCE
            iteration(int): Current iteration, needed for fixed RNG

        Returns:
            tuple[int, tuple[int]]: A tuple containing the current SCE index (for multiprocessing reasons) and the solution path
        """

        # due to multiprocessing sharing the RNG,
        # each function call needs its own individual seed.
        # Using the current iteration and sce_index makes it reproducible.
        # Otherwise, random bits from system entropy are used to init the seed, creating "more" randomness
        if self._fixed_rng:
            random.seed(sce_index*iteration+iteration)
        else:
            random.seed(int.from_bytes(urandom(4), byteorder='little'))
        # start_node = random.randrange(0, self.problem.dimension)

        # create list of unvisited nodes, remove first random node (called set U in paper)
        # solution, unvisited = [start_node], list(range(0, self.problem.dimension))
        # unvisited.remove(start_node)

        # TODO probability to choose between previous solution, previous best solution and random solution

        best_solution = self._tree.get_best_solution(self._tree.tree.root)

        # print(sce_index, linear_div_inflation)
        # print(linear_div_inflation)

        # TODO pass different types of variables (like continuous or categorical) in populations

        search_parameter = {'problem': self._problem,
                            'sce_index': sce_index,
                            'populations': self._tree.get_populations(sce_index),
                            'w_personal_previous': self._w_pers_prev,
                            'w_personal_best': self._w_pers_best,
                            'w_parent_best': self._w_parent_best,
                            'best_solution': best_solution,
                            'solution_creating_behavior': self._solution_creating_behaviour,
                            'linear_div_inflations': self._linear_div_inflations,
                            'h_probabilities': self._h_probabilities,
                            }

        # get the solution populations from the tree as dict
        # populations = self.tree.get_populations(sce_index)
        # print("in HSPPBO_TISD" + " " + str(populations))

        # first node already present, therefore subtract 1
        # for i in range(0, self.problem.dimension-1):

            # create array of tau values for each unvisited path i,k with k as all unvisited nodes
            # tau_arr = np.array(
            #    [self.tau(populations, solution[i], k) for k in tuple(unvisited)])

            # create probability distribution from the tau array and select a new node based on that
            # rnd_distr = tau_arr / np.sum(tau_arr)
            # next_node = random.choices(unvisited, rnd_distr)[0]

        # linear_deviation_in_i_th_dimension = [abs(row[i] - means_row_vector[i]) for row in matrix]

        # sum_of_linear_deviations = reduce(lambda a, b: a + b, linear_deviation_in_i_th_dimension)

        # inflated linear deviation
        # linear_div_inflated[i] = linear_div_inflation * (1 / (number_of_rows - 1)) * sum_of_linear_deviations

            # add new node to the solution path and remove it from the unvisited list
            # solution.append(next_node)
            # unvisited.remove(next_node)

        # random_distribution = [1-(self.w_pers_prev + self.w_pers_best + self.w_parent_best),
        #                       self.w_pers_prev, self.w_pers_best, self.w_parent_best]

        # print(random_distribution)

        # random_solution = self.problem.init_solution()[0]

        # print(random_solution)

        # possible_solutions = (random_solution,
        #                      populations[0],
        #                      populations[1],
        #                      populations[2])

        # next_solution = random.choices(possible_solutions, random_distribution)

        # TODO muss früher passieren
        additionally_so_far = 0

        # print("in HSSPBO_TISD find_next_solution: " + str(self._tree.get_solution(0)))

        # for sce_index in range(self._sce_count):

            # print("in HSPPBO_TISD find_next_solutoin: " + str(sce_index))
        #    additionally_so_far += self._tree.get_solution(sce_index)[0]["so_far"]

        # print("in HSPPBO_TISD find_next_solution: " + str(additionally_so_far))

        next_solution = 0

        # print('in HSPPBO: ' + 'test')

        # if self._problem.type == 'TISD':

            # print('test')

        #    integrand_sign = False

        #    while integrand_sign == False:

        #        solution_and_integrand = self._search_algorithm.construct_solution(self, search_parameter)

        #        integrand_sign = solution_and_integrand[1]

        #        print('in HSPPBO construct_solution: ' + str(integrand_sign))

                # print(next_solution[0])

                # next_solution_quality = self.problem.get_solution_quality(next_solution[0])

        #        next_solution = solution_and_integrand[0]

        # else:

        #    next_solution = self._search_algorithm.construct_solution(self, search_parameter)



        condition = False

        match self._accepted_solutions:

            case 'exact-term':

                while not condition:

                    additionally_so_far += 1

                    next_solution = self._search_algorithm.construct_solution(self, search_parameter)
                    condition = self._problem.get_solution_quality(next_solution)[1]

                self._problem.add_req_iterations(additionally_so_far)

                next_solution["so_far"] = self._problem.req_iterations()

            case 'penalty-term':

                next_solution = self._search_algorithm.construct_solution(self, search_parameter)
                next_solution["so_far"] = 0
                # print("in HSPPBO_TISD find_next_solution: " + str(next_solution))

            case _:

                pass



        # next_solution = self._search_algorithm.construct_solution(self, search_parameter)

        # print(f"in SHPPBO {next_solution}")

        # test = TISD(tisd_name='tisd10')
        # test_solution = test.init_solution()

        # print(solution)
        # print(tuple(solution))
        # return sce_index, tuple(solution)
        # print("hsppbo_tisd find_next_solution: " + str(next_solution))

        return [sce_index, next_solution]

    def tau(self, populations: tuple, i: int, k: int) -> float:
        """
        Calculuates tau_(ik) given the population (solution paths), and taking into account the heuristic,
        to the power of alpha and beta respectively.

        Args:
            populations (tuple): the populations (aka solution paths) of a SCE node
            i (int): the current node/state of the SCE
            k (int): the potential next node/state of the SCE

        Returns:
            float: the tau value for the given i,k
        """
        pop_range_sum = ((self._w_pers_prev if self.is_solution_subset((i, k), populations[0]) else 0)
                         + (self._w_pers_best if self.is_solution_subset((i, k), populations[1]) else 0)
                         + (self._w_parent_best if self.is_solution_subset((i, k), populations[2]) else 0))

        return pow(self._w_rand + pop_range_sum, self._alpha) * pow(self._problem.get_heuristic_component(i, k), self._beta)

    @staticmethod
    def is_solution_subset(subset: tuple, solution: dict) -> bool:
        """
        Checks whether the provided subset is an ordered and concurrent part of the provided solution.
        That means gaps within the match of the subset are not allowed.
        Keep in mind, that the solution has to be a dictionary, with the nodes as key and index in the list as value,
        so each entry is (node: index)

        Reason: Dicts can be searched in O(1), compared to lists/tuples in O(n)

        Example:
        solution    subset
        [1,2,3,4]   [1,2]  -> TRUE
        [1,2,3,4]   [1,3]  -> FALSE

        Args:
            subset (tuple[int]): List of the subset to be matched for
            solution (tuple[int]): List of the solution to be matched against

        Returns:
            bool: True if the subset is contained within the solution as stated
        """
        try:
            return solution.get(subset[0]) + 1 == solution.get(subset[1])
        except:
            return False

    def get_info(self) -> dict:
        """
        Get information about the hsppbo parameters and its problem instace and SCE tree

        Returns:
            dict: info about the current hsppbo run
        """
        return {
            'hsppbo': {
                'max_iteration_count': self._max_iteration_count,
                'w_pers_best': self._w_pers_best,
                'w_pers_prev': self._w_pers_prev,
                'w_parent_best': self._w_parent_best,
                'w_rand': self._w_rand,
                'alpha': self._alpha,
                'beta': self._beta,
                'detection_threshold': self._detection_threshold,
                'detection_pause': self._detection_pause,
                'reaction_type': self._reaction_type,
                'fixed_rng': self._fixed_rng
            },
            'tree': self._tree.get_info(),
            'problem': self._problem.get_info()
        }