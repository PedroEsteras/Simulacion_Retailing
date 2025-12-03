# This code was adapted from the source code from the paper:
#      Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.
# It is part of the following paper:
#      PENDING_PAPER_REFERENCE_TO_BE_ADDED
from math import sqrt

import numpy as np
from numpy import array
from sklearn.model_selection import KFold
from src.models.estimation import Estimator
from src.models.optimization.non_linear import NonLinearProblem, NonLinearSolver
from src.models.profiler import Profiler


class MaximumLikelihoodEstimator(Estimator):
    def estimate(self, model, transactions):
        problem = MaximumLikelihoodNonLinearProblem(model, transactions)
        solution = NonLinearSolver.default().solve(problem, self.profiler())
        model.update_parameters_from_vector(solution)
        return model


class MaximumLikelihoodNonLinearProblem(NonLinearProblem):
    def __init__(self, model, transactions):
        self.model = model
        self.transactions = transactions

    def constraints(self):
        return self.model.constraints()

    def objective_function(self, parameters):
        self.model.update_parameters_from_vector(parameters)
        return -self.model.log_likelihood_for(self.transactions)

    def amount_of_variables(self):
        return len(self.model.parameters_vector())

    def initial_solution(self):
        return array(self.model.parameters_vector())


class RegularizedMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _estimate_model_and_get_score(self, reg_factor, model, transactions):
        if (reg_factor < 0) or (reg_factor > 1):
            return 1e10
        elif reg_factor in self._cache.keys():
            return self._cache[reg_factor]
        else:
            kf = KFold(n_splits=3)
            score_per_fold = []

            for train_indexes, test_indexes in kf.split(transactions):
                train_trans = [transactions[i] for i in train_indexes]
                test_trans = [transactions[i] for i in test_indexes]

                tentative_model = model.clone()
                tentative_model.update_regularization_factor(reg_factor)

                problem = MaximumLikelihoodNonLinearProblem(tentative_model, train_trans)
                solution = NonLinearSolver.default().solve(problem, Profiler()) # use a new profiler
                tentative_model.update_parameters_from_vector(solution)
                score = tentative_model.log_likelihood_for(test_trans)
                score_per_fold += [score]

            avg_score = np.average(score_per_fold)
            self._cache[reg_factor] = avg_score
            return avg_score

    def _tune_regularization_factor_with_cross_validation(self, model, transactions, lower_limit=1e-4, upper_limit=1,
                                                          tolerance=1e-2):
        self._cache = {}
        # use golden search to find the best regularization factor based on 5-fold cross validation
        phi = (1 + sqrt(5)) * 0.5
        x1 = lower_limit
        x2 = upper_limit
        x3 = (x2 + x1 * phi) / (1 + phi)
        x4 = x1 + (x2 - x3)
        diff = (x2 - x1) / (x3 + x4)
        counter = 0
        while (diff > tolerance) and (counter <= 100):
            counter += 1

            score_x3 = self._estimate_model_and_get_score(x3, model, transactions)
            score_x4 = self._estimate_model_and_get_score(x4, model, transactions)

            if score_x4 < score_x3:
                x2 = x4
            else:
                x1 = x3
            x3 = (x2 + x1 * phi) / (1 + phi)
            x4 = x1 + (x2 - x3)
            diff = (x2 - x1) / (x3 + x4)

        reg_factor = x3
        return reg_factor


    def estimate(self, model, transactions):
        reg_factor = self._tune_regularization_factor_with_cross_validation(model, transactions)
        model.update_regularization_factor(reg_factor)
        self.reset_profiler() # reset the profiler to ensure enough iterations to calibrate the final model
        return super().estimate(model, transactions)
