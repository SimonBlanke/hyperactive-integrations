# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.base import BaseEstimator
from hyperactive import Hyperactive
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from sklearn.metrics import check_scoring

from .objective_function_wrapper import ObjectiveFunctionWrapper


class HyperactiveSearchCV(BaseEstimator):
    def __init__(self, estimator, hyperactive_tuner, params_config):
        self.estimator = estimator
        self.params_config = params_config

    def fit(self, X, y):
        objective_function = ObjectiveFunctionWrapper(
            self.estimator
        ).objective_function

        hyper = Hyperactive()
        hyper.add_search(objective_function, search_space=self.params_config)
        hyper.run()

    def score(self, X, y):
        pass
