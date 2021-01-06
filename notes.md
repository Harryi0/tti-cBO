# Notes for the L48 Project

## Possible inputs that vary across different strategies

**Boolean:**

* isolate_individual_on_symptoms
* isolate_individual_on_positive
* isolate_household_on_symptoms
* isolate_household_on_positive
* isolate_contacts_on_symptoms
* isolate_contacts_on_positive
* test_contacts_on_positive
* do_symptom_testing
* do_manual_tracing
* do_app_tracing

**Discrete**

* max_contacts

**Probability**

* go_to_school_prob
* wfh_prob
* compliance
* app_cov

**Other Possible input**

* testing_delay=2,  # Days delay between test and results
* manual_trace_delay=1,  # Delay associated with tracing manually

## Important outputs including the resource that might have constraint

* RETURN_KEYS.base_r, "Base R"
* RETURN_KEYS.reduced_r, "Effective R"

* RETURN_KEYS.tests, "# Tests Needed"
* RETURN_KEYS.man_trace, "# Manual Traces"
* RETURN_KEYS.app_trace, "# App Traces"
* RETURN_KEYS.quarantine, "# PersonDays Quarantined" 

## TODO: 

* [x] Create a target function with desired input and output for the tti-explorer

* [x] At first, test constrained bayesian optimization with constraint on total test numbers for  optimize the 
effective R value
    - [x] BO and cBO with GpyOpt: test cBO on other function, test BO with ttiModlel
        > <u>problem</u>: GpyOpt does not suppport constraint with a callable function
    - [x] BO and cBO with Emukit: test BO with ttiModel
        - [x] adding constraint for the ParameterSpace, but met problem during the optimization step for BO
        - [x] Try cBO with UnknownConstraintBayesianOptimizationLoop that would define the constraint as another 
        function emulated with e.g. GP
            1. Bayesian Optimization with Unknown Constraints: https://arxiv.org/pdf/1403.5607.pdf
            2. Pull request for UnknownConstraintBayesianOptimizationLoop: https://github.com/EmuKit/emukit/pull/217/files
    - [ ] Some Problems: how to choose the right kernel function? Non-negative value can not be well-modeled by a GP prior? why 


* [ ] Test cBO with other single constraint: **number of Manual Traces**

* [ ] Test cBO with other single constraint: **number of PersonDays Quarantined**

* [ ] Test cBO with multiple constraint (?): 

* [ ] Design a cost function as a constraint (?) on the input variables

* [ ] What about the noise? 

                                                                             