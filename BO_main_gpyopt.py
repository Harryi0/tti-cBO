import scipy
import GPyOpt
import GPy
import numpy as np
import pandas as pd
import time

from GPyOpt import experiment_design
from simulation_function import target_function

# input = {'isolate_individual_on_symptoms': 1,
#          'isolate_individual_on_positive': 1,
#          'isolate_household_on_symptoms': 1,
#          'isolate_household_on_positive': 1,
#          'isolate_contacts_on_symptoms': 0,
#          'isolate_contacts_on_positive': 1,
#          'test_contacts_on_positive': 1,
#          'do_symptom_testing': 1,
#          'do_manual_tracing': 1,
#          'do_app_tracing': 1,
#          'max_contacts': 10,
#          'app_cov': 0.35,
#          'compliance': 0.8,
#          'go_to_school_prob': 0.5,
#          'wfh_prob': 0.45
#          }
if __name__ == "__main__":
    space = [{'name': 'isolate_individual_on_symptoms', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'isolate_individual_on_positive', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'isolate_household_on_symptoms', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'isolate_household_on_positive', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'isolate_contacts_on_symptoms', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'isolate_contacts_on_positive', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'test_contacts_on_positive', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'do_symptom_testing', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'do_manual_tracing', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'do_app_tracing', 'type': 'discrete', 'domain': (0, 1)},
             {'name': 'max_contacts', 'type': 'discrete', 'domain': (1, 4, 10, 20, 2000)},
             {'name': 'app_cov', 'type': 'continuous', 'domain': (0, 1)},
             {'name': 'compliance', 'type': 'continuous', 'domain': (0, 1)},
             {'name': 'go_to_school_prob', 'type': 'continuous', 'domain': (0, 1)},
             {'name': 'wfh_prob', 'type': 'continuous', 'domain': (0, 1)}]

    keys = [item['name'] for item in space]
    var_types = [item['type'] for item in space]
    func = target_function(keys, var_types)



    # feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)
    feasible_region = GPyOpt.Design_space(space=space)

    # --- CHOOSE the intial design
    from numpy.random import seed  # fixed seed

    seed(123456)

    initial_design = experiment_design.initial_design('random', feasible_region, 10)

    # func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()


    objective = GPyOpt.core.task.SingleObjective(func.f)
    # objective = GPyOpt.core.task.SingleObjective(target_function)

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
                                                    initial_design)

    # --- Stop conditions
    max_time = None
    max_iter = 5
    tolerance = 1e-8  # distance between two consecutive observations

    # Run the optimization
    start_time = time.time()
    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
    end_time = time.time()
    print("Bayesian Optimization finished, total time {}".format(end_time-start_time))
    x_op = bo.x_opt
    y_op = bo.fx_opt
    main_output = func.main_output
    other_output = func.other_output

    outputs = pd.DataFrame({"Effective_R": main_output, "Total_Tests": other_output})

    # print("Total test number for each iteration: {}".format(other_output))
    # print("Effective R for each iteration: {}".format(main_output))

    print(outputs.round(2))
    print(x_op)
    print(y_op)
