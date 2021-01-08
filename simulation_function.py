import os

import time
import numpy as np
import pandas as pd

from tti_explorer import config, utils
from tti_explorer.case import simulate_case, CaseFactors
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import RETURN_KEYS

from tti_explorer.strategies.sample_delve import TTISampleModel

def print_doc(func):
    print(func.__doc__)

def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")

def get_complement_dictionary(dict1, dict2):
    return {k: dict1[k] for k in dict1 if k not in dict2}


class target_function:
    def __init__(self, keys, var_types, n_cases=1000, constraint_variable='# Tests Needed',
                 constraint_on_test=90, constraint_on_trace=90, constraint_on_quarantine=300):
        self.keys = keys
        self.var_types = var_types
        self.dims = len(self.keys)
        self.result_cache = dict()
        self.cache_used = 0

        #------------------------- set up constraint variable and constraint value ------
        self.target_variable = "Effective R"
        self.constraint_variable = constraint_variable
        # ["Base R", "# Manual Traces", "# Tests Needed","# PersonDays Quarantined"]
        self.constraint_value = {
            "# Tests Needed": constraint_on_test,
            "# Manual Traces": constraint_on_trace,
            "# PersonDays Quarantined": constraint_on_quarantine
        }
        self.constraint_on_test_num = constraint_on_test
        self.constraint_on_trace_num = constraint_on_trace
        self.constraint_on_quarantine = constraint_on_quarantine
        self.n_cases = n_cases

        #------------------------- Variable used in the TTIModel -------------------------
        path_to_bbc_data = os.path.join("data", "bbc-pandemic")
        self.over18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_o18.csv"))
        self.under18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_u18.csv"))
        self.rng = np.random.RandomState(42)
        name = 'S3_test_based_TTI_test_contacts'
        self.case_config = config.get_case_config("delve")
        self.contacts_config = config.get_contacts_config("delve")
        self.policy_config = config.get_strategy_configs("delve", name)[name]
        self.strategy_config = utils.get_sub_dictionary(self.policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
        # scale factor to turn simulation numbers into UK population numbers
        self.nppl = self.case_config['infection_proportions']['nppl']

    def f(self, x):
        input = dict()
        for i, k in enumerate(self.keys):
            if self.var_types[i] == 'discrete':
                input[k] = int(x[0,i])
            else:
                input[k] = x[0, i]

        res = self.target(**input)

        if res[1] > 70:
            res[0] = 3

        self.main_output.append(res[0])
        self.other_output.append(res[1])
        return res[0]

    def f_multi(self, X):
        Y = np.zeros((len(X),1))
        for n in range(len(X)):
            x = X[n,:]
            if tuple(x) in self.result_cache:
                print("cache used! ")
                self.cache_used += 1
                Y[n,:] = self.result_cache[tuple(x)][0]
            else:
                input = dict()
                for i, k in enumerate(self.keys):
                    input[k] = x[i]
                res = self.target(**input)
                self.main_output.append(res[0])
                self.other_output.append(res[1])
                Y[n,:] = res[0]
                self.result_cache[tuple(x)] = (res[0],res[1])
        return Y

    def c1(self, x):
        input = dict()
        if tuple(x) in self.result_cache:
            print("cache used! ")
            self.cache_used += 1
            c1 = self.result_cache[tuple(x)][1]
        else:
            for i, k in enumerate(self.keys):
                input[k] = x[i]
            res = self.target(**input)
            c1 = res[1]
            self.result_cache[tuple(x)] = (res[0], res[1])
        return c1

    def f_withSingleConst(self, X):
        '''
        target function wrapper on TTIModel return effective R with single constraint on one of
        ["# Manual Traces", "# Tests Needed","# PersonDays Quarantined"]
        :param X: Input variable dimension [num_points, num_input_variables]
        :return: [Y, Y_constraint]
        '''
        Y = np.zeros((len(X),1))
        Y_c = np.zeros((len(X),1))
        for n in range(len(X)):
            x = X[n,:]
            if tuple(x) in self.result_cache:
                print("TTIModel Results Cache used! ")
                self.cache_used += 1
                Y[n,:] = self.result_cache[tuple(x)][self.target_variable]
                Y_c[n,:] = self.result_cache[tuple(x)][self.constraint_variable] - \
                           self.constraint_value[self.constraint_variable]
            else:
                input = dict()
                for i, k in enumerate(self.keys):
                    input[k] = x[i]
                res = self.target(**input)
                Y[n,:] = res[self.target_variable]
                Y_c[n, :] = res[self.constraint_variable] - self.constraint_value[self.constraint_variable]
                self.result_cache[tuple(x)] = res
        return Y, Y_c

    def logf_withSingleConst(self, X):
        '''
        target function wrapper on TTIModel return effective R with single constraint on one of
        ["# Manual Traces", "# Tests Needed","# PersonDays Quarantined"]
        :param X: Input variable dimension [num_points, num_input_variables]
        :return: log value for [Y, Y_constraint]
        '''
        Y = np.zeros((len(X),1))
        Y_c = np.zeros((len(X),1))
        for n in range(len(X)):
            x = X[n,:]
            if tuple(x) in self.result_cache:
                print("TTIModel Results Cache used! ")
                self.cache_used += 1
                Y[n,:] = np.log(self.result_cache[tuple(x)][self.target_variable])
                Y_c[n,:] = np.log(self.result_cache[tuple(x)][self.constraint_variable])-\
                           np.log(self.constraint_value[self.constraint_variable])
            else:
                input = dict()
                for i, k in enumerate(self.keys):
                    input[k] = x[i]
                res = self.target(**input)
                Y[n,:] = np.log(res[self.target_variable])
                Y_c[n, :] = np.log(res[self.constraint_variable]) - \
                            np.log(self.constraint_value[self.constraint_variable])
                self.result_cache[tuple(x)] = res
        return Y, Y_c

    def target(
            self,
            max_contacts,
            test_contacts_on_positive=1,
            go_to_school_prob=0.5,
            wfh_prob=0.65,
            isolate_individual_on_symptoms=1,  # Isolate the individual after they present with symptoms
            isolate_individual_on_positive=1,  # Isolate the individual after they test positive
            do_symptom_testing=1,  # Test symptomatic individuals
            do_manual_tracing=1,  # Perform manual tracing of contacts
            do_app_tracing=1,  # Perform app tracing of contacts
            app_cov=0.35,  # Probability of tracing contact through app
            compliance=0.8,
            isolate_household_on_symptoms=1,  # Isolate the household after individual present with symptoms
            isolate_household_on_positive=1,  # Isolate the household after individual test positive
            isolate_contacts_on_symptoms=1,  # Isolate the contacts after individual present with symptoms
            isolate_contacts_on_positive=1,  # Isolate the contacts after individual test positive
            testing_delay=2,
            manual_trace_delay=1
    ):
        factor_config = {
            "app_cov": app_cov,
            "compliance": compliance,
            "go_to_school_prob": go_to_school_prob,
            "wfh_prob": wfh_prob
        }

        # input_strategy_config = utils.get_sub_dictionary(policy_config, config.INPUT_FACTOR_KEYS)
        input_strategy_config = {
            'isolate_individual_on_symptoms': int(isolate_individual_on_symptoms),
            'isolate_individual_on_positive': int(isolate_individual_on_positive),
            'isolate_household_on_symptoms': int(isolate_household_on_symptoms),
            'isolate_household_on_positive': int(isolate_household_on_positive),
            'isolate_contacts_on_symptoms': int(isolate_contacts_on_symptoms),
            'isolate_contacts_on_positive': int(isolate_contacts_on_positive),
            'test_contacts_on_positive': int(test_contacts_on_positive),
            'do_symptom_testing': int(do_symptom_testing),
            'do_manual_tracing': int(do_manual_tracing),
            'do_app_tracing': int(do_app_tracing),
            'max_contacts': int(max_contacts),
            'app_cov': int(app_cov),
            'compliance': int(compliance),
            'testing_delay': int(testing_delay),
            'manual_trace_delay': int(manual_trace_delay)
        }


        # fixed_strategy_config = utils.get_sub_dictionary(self.strategy_config, config.FIXED_FACTOR_KEYS)

        fixed_strategy_config = get_complement_dictionary(self.strategy_config, input_strategy_config)

        simulate_contacts = EmpiricalContactsSimulator(self.over18, self.under18, self.rng)

        tti_model = TTISampleModel(self.rng, **input_strategy_config, **fixed_strategy_config)

        outputs = list()

        # reduced_r, tests, 'man_trace', 'quarantine', 'base_r'
        start = time.time()
        for _ in range(self.n_cases):
            case = simulate_case(self.rng, **self.case_config)
            case_factors = CaseFactors.simulate_from(self.rng, case, **factor_config)
            contacts = simulate_contacts(case, **self.contacts_config)
            res = tti_model(case, contacts, case_factors)
            outputs.append(res)
        end = time.time()
        print("Finish iterations in the TTIModel! use time {}".format(end-start))

        # outputs = np.array(outputs)
        # This cell is mosltly just formatting results...

        # ["Base R", "# Manual Traces", "# App Traces", "# Tests Needed","# PersonDays Quarantined"]

        to_show = [
            RETURN_KEYS.reduced_r,
            RETURN_KEYS.tests,
            RETURN_KEYS.man_trace,
            RETURN_KEYS.quarantine,
            RETURN_KEYS.base_r,
        ]

        scales = []
        for show in to_show:
            if show == "Base R" or show == "Effective R":
                scales.append(1)
            else:
                scales.append(self.nppl)

        results = pd.DataFrame(outputs).mean(0).loc[to_show].mul(scales)
            # to_frame(name=f"Simulation results")
            # rename(index=lambda x: x + " (k per day)" if x.startswith("#") else x)

        # final_result = results.iloc[:, 0].values
        # print(results.round(1))
        return results.to_dict()
    #
    # n_iterations = 5
    #
    # x_init = np.random.rand(5, 1)
    # y_init, y_constraint_init = f(x_init)
    #
    # # Make GPy objective model
    # gpy_model = GPy.models.GPRegression(x_init, y_init)
    # model = GPyModelWrapper(gpy_model)
    #
    # # Make GPy constraint model
    # gpy_constraint_model = GPy.models.GPRegression(x_init, y_init)
    # constraint_model = GPyModelWrapper(gpy_constraint_model)
    #
    # space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    # acquisition = ExpectedImprovement(model)
    #
    # # Make loop and collect points
    # bo = UnknownConstraintBayesianOptimizationLoop(model_objective=model, space=space, acquisition=acquisition,
    #                                                model_constraint=constraint_model)
    # bo.run_loop(UserFunctionWrapper(f, extra_output_names=['Y_constraint']),
    #             FixedIterationsStoppingCondition(n_iterations))
    #
    # # Check we got the correct number of points
    # assert bo.loop_state.X.shape[0] == n_iterations + 5