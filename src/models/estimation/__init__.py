# This code was adapted from the source code from the paper:
#      Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.
# It is part of the following paper:
#      PENDING_PAPER_REFERENCE_TO_BE_ADDED

from src.models.profiler import Profiler


class Estimator(object):
    """
        Estimates a model parameters based on historical transactions data.
    """
    def __init__(self):
        self._profiler = Profiler()

    def profiler(self):
        return self._profiler

    def estimate(self, model, transactions):
        raise NotImplementedError('Subclass responsibility')

    def reset_profiler(self):
        self._profiler = Profiler()