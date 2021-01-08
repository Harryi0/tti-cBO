import scipy
import GPyOpt
import GPy
import numpy as np
import pandas as pd
import time

from GPyOpt import experiment_design
from emukit.bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
from emukit.core.initial_designs import RandomDesign
from emukit.core.loop import FixedIterationsStoppingCondition

from simulation_function import target_function

from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace, DiscreteParameter
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.constraints import NonlinearInequalityConstraint
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core.optimization import GradientAcquisitionOptimizer

if __name__ == "__main__":
    space = [
        {'name': 'isolate_individual_on_symptoms', 'type': 'discrete', 'domain': (0, 1)},
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
        # {'name': 'app_cov', 'type': 'discrete', 'domain': (0.35, 0.55, 0.75)},
        # {'name': 'compliance', 'type': 'discrete', 'domain': (0.8, 1)},
        {'name': 'go_to_school_prob', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'wfh_prob', 'type': 'continuous', 'domain': (0.25, 0.95)}
    ]

    keys = [item['name'] for item in space]
    var_types = [item['type'] for item in space]

    # constraint_variable from one of ["# Manual Traces", "# Tests Needed"]
    func = target_function(keys, var_types, n_cases=10000, constraint_variable="# Tests Needed", constraint_on_test=90)
    constraint_value = func.constraint_value[func.constraint_variable]

    use_log_function = True

    # constraint_1 = NonlinearInequalityConstraint(func.c1, lower_bound=None, upper_bound=np.array([90]))

    emukit_parameters = []
    for input in space:
        if input['type'] == 'discrete':
            emukit_parameters.append(DiscreteParameter(input['name'], list(input['domain'])))
        else:
            emukit_parameters.append(ContinuousParameter(input['name'], *list(input['domain'])))

    # emukit_space = ParameterSpace(emukit_parameters, [constraint_1])
    emukit_space = ParameterSpace(emukit_parameters)

    design = RandomDesign(emukit_space)

    n_initial_points = 10
    X_initial = design.get_samples(n_initial_points)

    ######### get Y with multi-dimensional x
    # Y_initial = func.f_multi(X_initial)
    ######### get Y and Y_c at the same time
    if use_log_function:
        Y_initial, Y_c_initial = func.logf_withSingleConst(X_initial)
    else:
        Y_initial, Y_c_initial = func.f_withSingleConst(X_initial)


    ######### gpy model with no unknow constraint
    # kern = GPy.kern.RBF(2, lengthscale=0.08, variance=20)
    # gpy_model = GPy.models.GPRegression(X_initial, Y_initial, kern, noise_var=1e-10)
    # emukit_model = GPyModelWrapper(gpy_model)
    ######### gpy model for objective function and the constraint function

    ######### Kernel selection based on marginal log likelihood
    kern_obj01 = GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0)
    kern_obj02 = GPy.kern.Linear(input_dim=X_initial.shape[1])
    kern_obj03 = ((GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0))+
                (GPy.kern.Linear(input_dim=X_initial.shape[1])))
    kern_obj04 = ((GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0))+
                (GPy.kern.Linear(input_dim=X_initial.shape[1]))+
                GPy.kern.White(input_dim=X_initial.shape[1], variance=0.1))
    allkerns_obj = [kern_obj01, kern_obj02, kern_obj03, kern_obj04]

    #--------------------- setup the GP model for objective function
    alllikelihood = np.zeros(len(allkerns_obj))
    for i, kernel in enumerate(allkerns_obj):
        alllikelihood[i] = GPy.models.GPRegression(X_initial, Y_initial, kernel=kernel).log_likelihood()
        print("Objective model: kernel No. {} with marginal loglikelihood {}".format(i+1, alllikelihood[i]))
    optimal_kern_idx = np.argmax(alllikelihood)
    print("Kernel No {} is the optimal for Objective function".format(optimal_kern_idx+1))
    gpy_model = GPy.models.GPRegression(X_initial, Y_initial, kernel=allkerns_obj[optimal_kern_idx])
    # gpy_model.Gaussian_noise.constrain_fixed(1e-4, warning=False)
    gpy_model.optimize()
    emukit_model = GPyModelWrapper(gpy_model)
    # Make GPy constraint model

    #--------------------- setup the GP model for constraint function
    kern_con01 = GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0)
    kern_con02 = GPy.kern.Linear(input_dim=X_initial.shape[1])
    kern_con03 = ((GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0))+
                (GPy.kern.Linear(input_dim=X_initial.shape[1])))
    kern_con04 = ((GPy.kern.RBF(input_dim=X_initial.shape[1], variance=1.0, lengthscale=1.0))+
                (GPy.kern.Linear(input_dim=X_initial.shape[1]))+
                GPy.kern.White(input_dim=X_initial.shape[1], variance=0.1))
    allkerns_con = [kern_con01, kern_con02, kern_con03, kern_con04]

    alllikelihood_constraint = np.zeros(len(allkerns_con))
    for i, kernel in enumerate(allkerns_con):
        alllikelihood_constraint[i] = GPy.models.GPRegression(X_initial, Y_c_initial, kernel=kernel).log_likelihood()
        print("Constraint model: kernel No. {} with marginal loglikelihood {}".format(i+1, alllikelihood_constraint[i]))
    optimal_kern_idx_constraint = np.argmax(alllikelihood_constraint)
    print("Kernel No {} is the optimal for constraint function".format(optimal_kern_idx_constraint+1))


    gpy_constraint_model = GPy.models.GPRegression(X_initial, Y_c_initial, kernel=allkerns_con[optimal_kern_idx_constraint])
    # gpy_constraint_model.Gaussian_noise.constrain_fixed(1e-4, warning=False)
    gpy_constraint_model.optimize()
    constraint_model = GPyModelWrapper(gpy_constraint_model)

    expected_improvement = ExpectedImprovement(emukit_model)


    max_iterations = 10
    ########## simple BO loop (with normal constraint)
    # bayesopt_loop = BayesianOptimizationLoop(emukit_space, emukit_model, acquisition=expected_improvement)
    #
    # bayesopt_loop.run_loop(func.f_multi, max_iterations)
    # results = bayesopt_loop.get_results()
    ########## BO loop with unknown constraint
    bayesopt_uncon_loop = UnknownConstraintBayesianOptimizationLoop(model_objective=emukit_model,
                                                                    space=emukit_space,
                                                                    acquisition=expected_improvement,
                                                                    model_constraint=constraint_model)
    if use_log_function:
        bayesopt_uncon_loop.run_loop(UserFunctionWrapper(func.logf_withSingleConst, extra_output_names=['Y_constraint']),
                                 FixedIterationsStoppingCondition(max_iterations))
        loopstate = bayesopt_uncon_loop.loop_state

        iters_X, iters_Y = loopstate.X, np.exp(loopstate.Y.flatten())
        iters_Y_c = np.array([np.exp(r.extra_outputs['Y_constraint'][0]) * constraint_value for r in loopstate.results])
    else:
        bayesopt_uncon_loop.run_loop(UserFunctionWrapper(func.f_withSingleConst, extra_output_names=['Y_constraint']),
                                 FixedIterationsStoppingCondition(max_iterations))
        loopstate = bayesopt_uncon_loop.loop_state

        iters_X, iters_Y = loopstate.X, loopstate.Y.flatten()
        iters_Y_c = np.array([r.extra_outputs['Y_constraint'][0] + constraint_value for r in loopstate.results])




    Y_Y_c = pd.DataFrame({'Y': iters_Y, 'Y_c': iters_Y_c})

    print(Y_Y_c.round(2))

    X_dict = dict()
    for i,k in enumerate(keys):
        X_dict[k] = iters_X[:,i]

    X_table = pd.DataFrame(X_dict)

    pd.set_option('display.max_columns', None)

    results = pd.concat([Y_Y_c, X_table], axis=1)
    results = results[['Y', 'Y_c']+keys]

    print(results)

    results_withConst = results[results.Y_c <= constraint_value]

    optimal_result = results_withConst.iloc[np.argmin(results_withConst.Y),:]

    # print("constrain on total test number below {}".format(func.constraint_on_test_num))
    # print("constrain on total manual trace number below {}".format(func.constraint_on_trace_num))
    print("single constrain on {} < {}".format(func.constraint_variable, constraint_value))

    print(optimal_result)

    print("finish cBO")



    # constraints = [{'name': 'constr_1', 'constraint': 'func.c(x)-50.0'}]

    # feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)
    # feasible_region = GPyOpt.Design_space(space=space)
    #
    # # --- CHOOSE the intial design
    # from numpy.random import seed  # fixed seed
    #
    # seed(123456)
    #
    # initial_design = experiment_design.initial_design('random', feasible_region, 10)
    #
    # # func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
    #
    #
    # objective = GPyOpt.core.task.SingleObjective(func.f)
    # # objective = GPyOpt.core.task.SingleObjective(target_function)
    #
    # model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)
    #
    # aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    #
    # acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
    #
    # evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    #
    # bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator,
    #                                                 initial_design)
    #
    # # --- Stop conditions
    # max_time = None
    # max_iter = 5
    # tolerance = 1e-8  # distance between two consecutive observations
    #
    # # Run the optimization
    # start_time = time.time()
    # bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
    # end_time = time.time()
    # print("Bayesian Optimization finished, total time {}".format(end_time-start_time))
    # x_op = bo.x_opt
    # y_op = bo.fx_opt
    # main_output = func.main_output
    # other_output = func.other_output
    #
    # outputs = pd.DataFrame({"Effective_R": main_output, "Total_Tests": other_output})
    #
    # # print("Total test number for each iteration: {}".format(other_output))
    # # print("Effective R for each iteration: {}".format(main_output))
    #
    # print(outputs.round(2))
    # print(x_op)
    # print(y_op)
