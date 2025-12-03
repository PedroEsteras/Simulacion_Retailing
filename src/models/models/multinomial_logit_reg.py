# This code was adapted from the source code from the paper:
#      Berbeglia, G., Garassino, A., & Vulcano, G. (2018). A comparative empirical study of discrete choice models in retail operations. Available at SSRN 3136816.
# It is part of the following paper:
#      PENDING_PAPER_REFERENCE_TO_BE_ADDED
from math import sqrt

from numpy import ones
from src.models.models import Model
from src.models.utils import generate_n_equal_numbers_that_sum_one, generate_n_random_numbers_that_sum_m, ZERO_LOWER_BOUND, \
    safe_log
from src.models.optimization.non_linear import Constraints
import src.models.settings as settings
from src.models.models import MultinomialLogitModel

class MultinomialLogitRegModel(MultinomialLogitModel):
    @classmethod
    def code(cls):
        return 'mnlreg'

    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['etas'], data['reg_factor'], data['norm_type'])

    @classmethod
    def simple_deterministic_w_l1reg(cls, products, l1factor):
        return cls(products, generate_n_equal_numbers_that_sum_one(len(products) - 1), l1factor, 1)

    @classmethod
    def simple_random_w_l1reg(cls, products, l1factor):
        return cls(products, generate_n_random_numbers_that_sum_m(len(products) - 1, 2), l1factor, 1)

    @classmethod
    def simple_deterministic_w_l2reg(cls, products, l2factor):
        return cls(products, generate_n_equal_numbers_that_sum_one(len(products) - 1), l2factor, 2)

    @classmethod
    def simple_random_w_l2reg(cls, products, l2factor):
        return cls(products, generate_n_random_numbers_that_sum_m(len(products) - 1, 2), l2factor, 2)


    def __init__(self, products, etas, reg_factor, norm_type=2):
        super(MultinomialLogitRegModel, self).__init__(products, etas)
        self.etas = etas
        self.reg_factor = reg_factor
        self.norm_type = norm_type
        if not(0 <= reg_factor <= 1):
            raise Exception("Regularization factor must be between 0 and 1")
        if norm_type not in [1, 2]:
            raise Exception("Only norm-1 and norm-2 types are supported for regularization")

    def log_probability_of(self, transaction):
        den = sum([self.utility_of(product) for product in transaction.offered_products])
        num  = safe_log(self.utility_of(transaction.product))
        return num - safe_log(den)

    def regularization_penalty(self):
        if self.norm_type == 1:
            l1_norm = sum(abs(safe_log(eta)) for eta in self.etas)
            reg = self.reg_factor * l1_norm
        elif self.norm_type == 2:
            l2_norm = sqrt(sum(pow(safe_log(eta), 2) for eta in self.etas))
            reg = self.reg_factor * l2_norm
        else:
            raise Exception("Only norm-1 and norm-2 types are supported for regularization")
        return reg

    def update_regularization_factor(self, new_reg_factor):
        if 0 <= new_reg_factor <= 1:
            self.reg_factor = new_reg_factor
        else:
            raise Exception("Regularization factor must be between 0 and 1")

    def data(self):
        return {
            'code': self.code(),
            'products': self.products,
            'etas': self.etas,
            'reg_factor': self.reg_factor,
            'norm_type': self.norm_type,
        }

    def __repr__(self):
        return '<Products: %s ; Etas: %s ;  Reg. factor:  %s ; Norm type:  %s >' % (self.products,
                                                                                    self.etas,
                                                                                    self.reg_factor,
                                                                                    self.norm_type)
