# This code was adapted from the source code from the paper:
#      Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.
# It is part of the following paper:
#      PENDING_PAPER_REFERENCE_TO_BE_ADDED

from src.models.estimation.expectation_maximization import ExpectationMaximizationEstimator
from src.models.estimation.ranked_list import RankedListEstimator


class RankedListExpectationMaximizationEstimator(ExpectationMaximizationEstimator, RankedListEstimator):
    def one_step(self, model, transactions):
        x = [[0 for _ in transactions] for _ in model.ranked_lists]

        for t, transaction in enumerate(transactions):
            compatibles = model.ranked_lists_compatible_with(transaction)
            den = sum([model.beta_for(compatible[0]) for compatible in compatibles])
            for i, ranked_list in compatibles:
                x[i][t] = model.beta_for(i) / den

        m = [sum(x[i]) for i in range(len(model.ranked_lists))]
        model.set_betas([m[i] / sum(m) for i in range(len(model.ranked_lists))])
        return model
