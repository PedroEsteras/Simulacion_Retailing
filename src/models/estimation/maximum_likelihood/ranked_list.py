# This code was adapted from the source code from the paper:
#      Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.
# It is part of the following paper:
#      PENDING_PAPER_REFERENCE_TO_BE_ADDED

from src.models.estimation.maximum_likelihood import MaximumLikelihoodEstimator
from src.models.estimation.ranked_list import RankedListEstimator


class RankedListMaximumLikelihoodEstimator(MaximumLikelihoodEstimator, RankedListEstimator):
    pass
