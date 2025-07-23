import os
# import sys
import argparse
from importlib import util, import_module
from math import floor

# from mpire.dashboard.dashboard import progress_bar_new

from hsppbo import HSPPBO
from hsppbo_tisd import HSPPBO_TISD
from logger import Logger
from optimizer import Optimizer
from mixed_variable_problems.artificial_mixed_variable_problems.cigar_mv import Cigar_MV
from mixed_variable_problems.artificial_mixed_variable_problems.ellipsoid_mv import Ellipsoid_MV
from mixed_variable_problems.artificial_mixed_variable_problems.paraboloid_mv import Paraboloid_MV
from mixed_variable_problems.artificial_mixed_variable_problems.tablet_mv import Tablet_MV
from mixed_variable_problems.csd_with_exact_term import CSD_exact
from mixed_variable_problems.csd_with_penalty_term import CSD_penalty
from combinatorial_problems.tsp import TSP
from mixed_variable_problems.tisd import TISD
from continuous_problems.simple_continuous_problems.cigar import Cigar
from continuous_problems.simple_continuous_problems.de_jong import De_Jong
from continuous_problems.simple_continuous_problems.ellipsoid import Ellipsoid
from continuous_problems.simple_continuous_problems.paraboloid import Paraboloid
from continuous_problems.simple_continuous_problems.plane import Plane
from continuous_problems.simple_continuous_problems.tablet import Tablet
from continuous_problems.b_2 import B_2
from continuous_problems.branin_rcos import Branin_RCOS
from continuous_problems.easom import Easom
from continuous_problems.goldstein_and_price import Goldstein_and_Price
from continuous_problems.griewangk import Griewangk
from continuous_problems.hartmann_3_4 import Hartmann_3_4
from continuous_problems.hartmann_6_4 import Hartmann_6_4
from continuous_problems.martin_and_gaddy import Martin_and_Gaddy
from continuous_problems.rosenbrock import Rosenbrock
from continuous_problems.shekel_4_5 import Shekel_4_5
from continuous_problems.shekel_4_7 import Shekel_4_7
from continuous_problems.shekel_4_10 import Shekel_4_10
from continuous_problems.zakharov import Zakharov
from config import params
import timeit

from src.continuous_problems.simple_continuous_problems.diagonal_plane import Diagonal_Plane


def user_input():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        help="Turn on output verbosity", action='store_true')
    parser.add_argument('-m', '--mode', type=str, choices=[
                        'run', 'opt', 'exp'], default='run', help='Mode of execution for the given problem and/or algorithm')
    parser.add_argument('-i', '--runs', type=int, default=50,
                        help='Number of runs to execute consecutively')
    parser.add_argument('-cr', '--continue-run', type=int, default=0,
                        help='Run number from which the calculations shall be continued (only opt mode)')
    parser.add_argument('-td', '--test-dynamic',
                        help="Set, if all dynamic intensities (set in the config for the respective mode) shall be tested", action='store_true')
    # parser.add_argument('-p', '--problem', type=str, default='rat195',
    #                    help='Name of the problem instance, e.g. the TSPLIB names like "rat195"')


    parser.add_argument('-p', '--problem', type=str, default='paraboloid_6',
                        help='Name of the problem instance, e.g. TISD names like "tisd10" or TSPLIB names like "rat195"')
    # parser.add_argument('-pt', '--problem-type', type=str, default='TSP',
    #                    help='Type of the problem, e.g. TSP (standard, symmetric TSP), ATSP (asymmetric TSP), QAP, TISD')
    parser.add_argument('-pt', '--problem-type', type=str, default='Paraboloid',
                        help='Type of the problem, e.g. TSP (standard, symmetric TSP), ATSP (asymmetric TSP), QAP, TISD')


    parser.add_argument('-rot', '--rotation', action=argparse.BooleanOptionalAction, default=False,
                        help='enables random rotation of the given test function')


    parser.add_argument('-salg', '--search_algorithm', type=str, default='h_het_p_aco_r_very_simple',
                        help='Type of the search algorithm, e.g. random_search or aco_r_very_simple')
    parser.add_argument('-salgn', '--search_algorithm_name', type=str, default='H_Het_P_ACO_R_Very_Simple',
                        help='Name of the search algorithm, e.g. Random_Search or ACO_R_Very_Simple')
    parser.add_argument('-mitc', '--max_iteration_count', type=int, default=10000,
                        help='Maximum number of iterations (not function evaluations) to run the search algorithm')
    parser.add_argument('-mfev', '--max_func_evaluations', type=int, default=8000,
                        help='Maximum number of function evaluations to run the search algorithm')


    # TODO implement accuracy
    parser.add_argument('-acc', '--accuracy', type=str, choices=['absolute', 'relative'], default='relative',
                        help='Accuracy at which the algorithm;'
                             'absolute: |f - f*| < Term, relative: |f - f*| < |Term * f| + Term')
    parser.add_argument('-absa', '--absolute_accuracy_term', type=float, default=10**(-10),
                        help='Absolute accuracy at which the algorithm;')
    parser.add_argument('-rela', '--relative_accuracy_term', type=float, default=10**(-4),
                        help='Relative accuracy at which the algorithm;')


    parser.add_argument('-rsb', '--restart_behaviour', type=str, choices=['fixed', 'absolute', 'adaptive'],
                        default='fixed', help='Algorithm restarts a fixed amount of times,'
                                                 'after an absolut number of iterations'
                                                 'or when the difference between solutions is under a certain threshold')
    parser.add_argument('-per', '--period', type=int, default=7,
                        help='Period at which the Algorithm checks if it should restart')
    parser.add_argument('-at', '--adaptive_threshold', type=float, default=10 ** (-10),
                        help='The difference between two solutions at which the algorithm restarts')
    parser.add_argument('-nor', '--number_of_fixed_restarts', type=int, default=0,
                        help='The number of fixed restarts the algorithm performs')
    parser.add_argument('-noi', '--number_of_iterations_per_run', type=int, default=8000,
                        help='The number of iterations per run')
    parser.add_argument('-po', '--problem_optimum', type=float, default=0,
                        help='the known optimum of the problem')
    parser.add_argument('-as', '--accepted_solutions', type=str, choices=['exact-term', 'penalty-term'], default='penalty-term',
                        help='The way initial solutions are calculated; exact-term means only initial solutions that fit the constrains'
                             'are allowed; penalty-term means all solutions are allowed, regardless if they fail the requirements')


    parser.add_argument('-scb', '--solution-creating-behaviour', type=str,
                        choices=['homogeneous', 'weak-heterogeneous', 'heterogeneous'], default='weak-heterogeneous',
                        help='')
    parser.add_argument('-ldi', '--linear_div_inflations', type=tuple, default=[0.7, 1.0, 1.3, 1.5])
    parser.add_argument('-hp', '--h-probabilities', type=tuple, default=[0.0, 0.05, 0.1, 0.15],
                        help='probabilities for each hierarchical level at which instead of a calculated solation a random solution is taken')


    parser.add_argument('-di', '--dynamic-intensity', type=check_percent_range,
                        default=0.25, help='Intensity of the dynamic problem instance')
    parser.add_argument('-opt', '--opt-algo', type=str, choices=[
                        'random', 'bayesian', 'forest', 'gradient'], default='bayesian', help='Algorithm used in optimization process')
    parser.add_argument('-oc', '--obj-calls', type=int, default=20,
                        help='Number of calls to the objective function during optimization. Not important otherwise.')
    parser.add_argument('-sc', '--sce_count', type=int, choices=[4, 13, 40], default=13,
                        help='Number of SCEs\' in the SCE-tree, 4, 13 or 40')
    parser.add_argument('-scc', '--sce_child_count', type=int, default=3,
                        help='Number of child-nodes per SCE')
    parser.add_argument('-plb', '--personal-best', type=check_percent_range, default=0.075,
                        help='Influence of the SCE\'s best personal solution ever obtained')
    parser.add_argument('-ppr', '--personal-previous', type=check_percent_range,
                        default=0.075, help='Influence of the SCE\'s previous personal solution')
    parser.add_argument('-ptb', '--parent-best', type=check_percent_range, default=0.01,
                        help='Influence of the SCE\'s parent personal best solution ever obtained')


    parser.add_argument('-a', '--alpha', type=int, default=1,
                        help='Influence of the pheromone/non-heuristic')
    parser.add_argument('-b', '--beta', type=int, default=7,
                        help='Influence of the heuristic')
    # parser.add_argument('-')
    parser.add_argument('-ddt', '--dynamic-detection-threshold', type=check_percent_range, default=0.25,
                        help='Threshold (swaps per SCE and iteration) for detecting a change')
    # TODO find out what detection pause does
    parser.add_argument('-dp', '--detection_pause', type=int, default=5,
                        help='')
    parser.add_argument('-r', '--reaction-type', type=str, choices=[
                        'partial', 'full', 'none'], default='partial', help='Type of reaction algorithm used to handle a change')

    args = parser.parse_args()

    # print(args.search_algorithms)

    return args

# TODO: implement pheromon buckshot

# TODO: implement absolute and relatve restart


# check if number is between 0 and 1
def check_percent_range(number: float) -> float:
    try:
        number = float(number)
    except ValueError:
        raise argparse.ArgumentTypeError('Number is no floating point literal')

    if 0.0 > number or number > 1.0:
        raise argparse.ArgumentTypeError('Number has to be between 0 and 1')

    return number


def custom_import(name, path):
    # Define the full path of the module
    module_path = path
    # Load the module from the given path
    spec = util.spec_from_file_location(name, module_path)
    module = util.module_from_spec(spec)
    # sys.modules[name] = module
    spec.loader.exec_module(module)
    # module = module.Random_Search()
    module = getattr(module, name)
    # print(module)
    # module = module.Random_Search()
    # spec.loader.exec_module(module)

    return module

# TODO code:
#   - all modes: add relative difference to optimal solution to run and exp output
#   - analyzer:
#       - comparing opt methods (convergence plot), wilcoxon signed rank test (https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
#       - comparing opt params (boxplots, partial dependence plot, feature importance)
#       - comparing exp runs (avg run analysis, alc precision/recall and roc curve for reset accuracy)


# TODO code(optional):
#   - implement tests
#   - web UI (flask) and better packaging

# TODO thesis:
#   - how does optimizer handle categorial values? -> onehot and label encoding (onehot used)

def main():
    args = user_input()

    logger = Logger(mode=args.mode)

    match args.problem_type:

        case 'TSP':
            problem = TSP(tsp_name=args.problem)

        case 'Cigar_MV':
            problem = Cigar_MV(artificial_mixed_variable_pr_name=args.problem)

        case 'Ellipsoid_MV':
            problem = Ellipsoid_MV(artificial_mixed_variable_pr_name=args.problem)

        case 'Paraboloid_MV':
            problem = Paraboloid_MV(artificial_mixed_variable_pr_name=args.problem)

        case 'Tablet_MV':
            problem = Tablet_MV(artificial_mixed_variable_pr_name=args.problem)

        case 'CSD_exact':
            problem = CSD_exact(csd_name=args.problem)

        case 'CSD_penalty':
            problem = CSD_penalty(csd_name=args.problem)

        case 'TISD':
            problem = TISD(tisd_name=args.problem)

        case 'Cigar':
            problem = Cigar(simple_con_pr_name=args.problem)

        case 'De_Jong':
            problem = De_Jong(simple_con_pr_name=args.problem)

        case 'Diagonal_Plane':
            problem = Diagonal_Plane(simple_con_pr_name=args.problem)

        case 'Ellipsoid':
            problem = Ellipsoid(simple_con_pr_name=args.problem)

        case 'Paraboloid':
            problem = Paraboloid(simple_con_pr_name=args.problem)

        case 'Plane':
            problem = Plane(simple_con_pr_name=args.problem)

        case 'Tablet':
            problem = Tablet(con_pr_name=args.problem)

        case 'B_2':
            problem = B_2(con_pr_name=args.problem)

        case 'Branin_RCOS':
            problem = Branin_RCOS(con_pr_name=args.problem)

        case 'Easom':
            problem = Easom(con_pr_name=args.problem)

        case 'Goldstein_and_Price':
            problem = Goldstein_and_Price(con_pr_name=args.problem)

        case 'Griewangk':
            problem = Griewangk(con_pr_name=args.problem)

        case 'Hartmann_3_4':
            problem = Hartmann_3_4(con_pr_name=args.problem)

        case 'Hartmann_6_4':
            problem = Hartmann_6_4(con_pr_name=args.problem)

        case 'Martin_and_Gaddy':
            problem = Martin_and_Gaddy(con_pr_name=args.problem)

        case 'Rosenbrock':
            problem = Rosenbrock(con_pr_name=args.problem)

        case "Shekel_4_5":
            problem = Shekel_4_5(con_pr_name=args.problem)

        case "Shekel_4_7":
            problem = Shekel_4_7(con_pr_name=args.problem)

        case "Shekel_4_10":
            problem = Shekel_4_10(con_pr_name=args.problem)

        case 'Zakharov':
            problem = Zakharov(con_pr_name=args.problem)

        case _:
            raise NotImplementedError(
                'Problem type not implemented yet')

    search_algorithm_type = args.search_algorithm
    search_algorithm_name = args.search_algorithm_name
    # search_algorithm_type = 'search_algorithms/random_search.py'
    path_to_search_pattern = 'search_algorithms/' + search_algorithm_type + '.py'
    # path_to_search_pattern = search_algorithm_type
    # print(path_to_search_pattern)
    # module = import_module('random_search.py', 'search_algorithms')
    module = custom_import(search_algorithm_name, path_to_search_pattern)
    # module = import_module('search_algorithms/random_search.py')
    # module = getattr(module, search_algorithm_name)
    # print(module)
    # my_class = my_import('test', path_to_search_pattern)
    # from module import Random_Search
    search_algorithm = module
    # test = module.Random_Search()
    # search_algorithms = getattr(module, 'Random_Search')

    # print(search_algorithms)

    # if (args.problem_type == 'TSP'):
    #    problem = TSP(tsp_name=args.problem)
    # else:
    #    raise NotImplementedError(
    #        'Problem type not implemented yet')

    # set wether the problem is dynamic or not
    if args.dynamic_intensity > 0:
        problem.set_dynamic(args.dynamic_intensity, min_iteration_count=2000-1)

    if args.restart_behaviour == 'fixed':

        args.max_iteration_count = round(args.max_iteration_count / (args.number_of_fixed_restarts + 1))

    if args.restart_behaviour == 'absolut':

        args.number_of_fixed_restarts = floor(args.max_iteration_count / args.number_of_iterations_per_run) - 1

    TSP_parameters = {'alpha': args.alpha,
                      'beta': args.beta
                      }

    SEC_parameters = {'sce_count': args.sce_count,
                      'sce_child_count': args.sce_child_count,
                      'w_personal_best': args.personal_best,
                      'w_personal_previous': args.personal_previous,
                      'w_parent_best': args.parent_best
                      }

    HSPPBO_parameters = {'problem': problem,
                         'logger': logger,
                         'search_algorithm': search_algorithm,
                         'restart_behaviour': args.restart_behaviour,
                         'period': args.period,
                         'adaptive_threshold': args.adaptive_threshold,
                         'accuracy': args.accuracy,
                         'absolute_acc': args.absolute_accuracy_term,
                         'relative_acc': args.relative_accuracy_term,
                         'problem_optimum': args.problem_optimum,
                         'SCE_parameters': SEC_parameters,
                         'TSP_parameters': TSP_parameters,
                         'accepted_solutions': args.accepted_solutions,
                         'solution_creating_behaviour': args.solution_creating_behaviour,
                         'linear_div_inflations': args.linear_div_inflations,
                         'h_probabilities': args.h_probabilities,
                         'dynamic_detection_threshold': args.dynamic_detection_threshold,
                         'detection_pause': args.detection_pause,
                         'reaction_type': args.reaction_type,
                         'max_iteration_count': args.max_iteration_count,
                         'max_func_evaluations': args.max_func_evaluations
                         }

    # max_iteration_count=2600
    # hsppbo = HSPPBO(problem, logger, args.personal_best, args.personal_previous, args.parent_best,
    #                args.alpha, args.beta, args.dynamic_detection_threshold, args.reaction_type, max_iteration_count=2)

    # hsppbo = HSPPBO_TISD(problem, search_algorithm, logger, args.personal_best, args.personal_previous, args.parent_best, args.alpha,
    #                     args.beta, args.dynamic_detection_threshold, args.reaction_type,
    #                     max_iteration_count=100000)

    hsppbo = HSPPBO_TISD(HSPPBO_parameters)

    logger.set_info(hsppbo.get_info())

    match args.mode:

        case 'run':

            for i in range(args.runs):

                logger.init_mode()
                best_solution = [[0, 10**100], 0]
                # best_solution[0][0]["so_far"] = 0
                # best_solution[0][0]: argument of the solution
                # best_solution[0][1]: solution quality
                # best_solution[1]   : number of iterations (not function evaluation)
                starttime = timeit.default_timer()

                match args.restart_behaviour:

                    case 'fixed':

                        for j in range(args.number_of_fixed_restarts + 1): #

                            hsppbo.reset_tree()
                            new_solution = hsppbo.execute(verbose=args.verbose)

                            # print("in main new_solution: " + str(new_solution[1]))

                            # print("in main best_solution: " + str(best_solution[1]))
                            # print("in main: " + str(args.max_iteration_count))
                            # temp = [best_solution[1]]
                            # temp = best_solution[1] + args.max_iteration_count

                            best_solution[1] = best_solution[1] + new_solution[1] # number of iterations

                            if new_solution[0][1] < best_solution[0][1]:

                                # print("in main: " + str(new_solution[0][1]))

                                # print("in main: " + str(new_solution[:1]))

                                best_solution[:1] = new_solution[:1] # replaces [0, 10**100] with the argument of the new solution and its function value

                            # match args.accuracy:

                            #    case 'absolut'

                    case 'adaptive':

                        hsppbo.reset_tree()
                        new_solution = hsppbo.execute(verbose=args.verbose)

                        # print("in main new_solution: " + str(new_solution[1]))

                        # print("in main best_solution: " + str(best_solution[1]))
                        # print("in main: " + str(args.max_iteration_count))
                        # temp = [best_solution[1]]
                        # temp = best_solution[1] + args.max_iteration_count

                        best_solution[1] = best_solution[1] + new_solution[1]

                        if new_solution[0][1] < best_solution[0][1]:
                            # print("in main: " + str(new_solution[0][1]))

                            # print("in main: " + str(new_solution[:1]))

                            best_solution[:1] = new_solution[:1]  # replaces [0, 10**100] with the argument of the new solution its function value

                        # while best_solution[1] <= args.max_iteration_count:

                        #    hsppbo.reset_tree()
                        #    new_solution = hsppbo.execute(verbose=args.verbose)
                        #    temp_solution = new_solution
                        #    best_solution[1] = best_solution[1] + new_solution[1]

                        #    if new_solution[0][1] < best_solution[0][1]:

                        #        best_solution[:1] = new_solution[:1]

                        #    match args.accuracy:

                        #        case 'absolute':

                        #            if abs(best_solution[0][1] - args.problem_optimum) < args.absolute_accuracy_term:

                         #               break

                         #       case 'relative':

                         #           # print('in HSPPBO tisd execute ' + 'test')

                         #           if (abs(best_solution[0][1] - args.optimum)
                         #                   < (abs(args.relative_accuracy_term * best_solution[0][1]) + args.relative_accuracy_term)):

                         #               break

                         #       case _:

                         #           pass

                    case _:

                        pass

                logger.close_run_logger()
                print("Solution:", best_solution)
                print('Required pre iterations:', best_solution[0][0]["so_far"])
                print('Required iterations:', best_solution[1])
                print("Required restarts", best_solution[0][0]["required_restarts"])
                print('Required total iterations:', best_solution[1] + best_solution[0][0]["so_far"])
                print('Required function evaluations:', best_solution[1] * 13 + best_solution[0][0]["so_far"])
                print("Solution:", best_solution[0][0])
                print("Solution quality:", best_solution[0][1])
                print("Total execution time:", timeit.default_timer() - starttime)
                print("Run number: " + str(i + 1))

                hsppbo.reset_problem()

                # best_solution[0][0]["so_far"] = 0
                # best_solution[0][0]["required_restarts"] = 0

                # performance_path_write = "/home/user/Schreibtisch/Masterarbeit/Masterarbeit_XF_OPT_META_Framework/XF-OPT-META/src/performance_data/paraboloid_10.txt"

                abs_path = os.path.dirname(os.path.abspath(__file__))

                # print(abs_path)

                with open(os.path.join(abs_path, "performance_data/{}".format(args.problem) + "_" + args.restart_behaviour + "_" + "continuous" + ".txt"), "a+") as performance_data:

                    # perform_data_string = str(best_solution[0][0]["required_restarts"]) + " " + str(best_solution[1] * 13 + best_solution[0][0]["so_far"]) + " " + str(best_solution[0][1]) + str("\n")
                    perform_data_string = str(best_solution[0][0]["required_restarts"]) + " " + str(best_solution[1]) + " " + str(best_solution[1] * 13) + " " + str(best_solution[0][1]) + " " + str(timeit.default_timer() - starttime) + " " + str(best_solution[0][0]['continuous'][0]) + str("\n") # for continuous problems

                    performance_data.write(perform_data_string)

                    if i == (args.runs - 1):
                        performance_data.write(str("\n"))

        case 'exp':

            if args.test_dynamic:
                dynamic_num = sum(
                    [len(p) for p in params['exp']['problem']]) - len(params['opt']['problem'])
                logger.init_dynamic(params['exp']['problem'], dynamic_num)
            else:
                dynamic_num = 1

            n_runs = args.runs if args.runs != 0 else get_run_number()

            for d in range(1, dynamic_num+1):
                if args.test_dynamic:
                    problem.set_dynamic(
                        dynamic_intensity_pct=params['exp']['problem'][0][d])
                logger.init_mode(n_runs)

                for n in range(1, n_runs+1):
                    print("---STARTING EXPERIMENTATION RUN " +
                          str(n) + "/" + str(n_runs) + " AND DYNAMIC CONF " + str(d) + "/" + str(dynamic_num) + "---")

                    hsppbo.execute(verbose=args.verbose)
                    hsppbo._tree.reset()
                    problem.reset()
                    logger.close_run_logger()
                    logger.add_exp_run(n)
                logger.create_exp_avg_run()
                if args.test_dynamic:
                    logger.next_dynamic(
                        d, {params['exp']['problem'][0][0]: params['exp']['problem'][0][d]})

        case 'opt':

            opt_algo = args.opt_algo
            hsppbo.set_random_seed()
            logger.set_info(hsppbo.get_info())

            if args.test_dynamic:
                dynamic_num = sum(
                    [len(p) for p in params['opt']['problem']]) - len(params['opt']['problem'])
                logger.init_dynamic(params['opt']['problem'], dynamic_num)
            else:
                dynamic_num = 1

            n_runs = args.runs if args.runs != 0 else get_run_number()
            opt = Optimizer(opt_algo, hsppbo.execute_wrapper, params['opt']['hsppbo'])
            for d in range(1, dynamic_num+1):
                if args.test_dynamic:
                    problem.set_dynamic(
                        dynamic_intensity_pct=params['opt']['problem'][0][d])

                logger.init_mode(params['opt']['hsppbo'], opt_algo)

                for n in range(1+args.continue_run, n_runs+1):
                    print("---STARTING OPTIMIZATION RUN " +
                          str(n) + "/" + str(n_runs) + " AND DYNAMIC CONF " + str(d) + "/" + str(dynamic_num) + "---")
                    opt_res = opt.run(verbose=args.verbose,
                                      n_calls=args.obj_calls, random_state=n)
                    logger.create_opt_files(opt_res, run=n)
                logger.create_opt_best_params()
                if args.test_dynamic:
                    logger.next_dynamic(
                        d, **{params['opt']['problem'][0][0]: params['opt']['problem'][0][d]})


def get_run_number() -> int:
    print('Do you want to perform multiple n_runs? [Y/N]')
    x = input()
    if x.lower() == 'y' or x.lower() == 'yes':
        print('How many n_runs do you want to execute? (max. 30)')
        n_runs = input()
        if 0 < int(n_runs) <= 30:
            return int(n_runs)
    return 1


if __name__ == '__main__':
    main()
