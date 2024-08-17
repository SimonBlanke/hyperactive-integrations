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
    def __init__(self, estimator, optimizer, params_config):
        self.estimator = estimator
        self.optimizer = optimizer
        self.params_config = params_config

    def fit(self, X, y):
        objective_function = ObjectiveFunctionWrapper(
            self.estimator,
            X=X,
            y=y,
        ).objective_function

        hyper = Hyperactive()
        hyper.add_search(
            objective_function,
            search_space=self.params_config,
            optimizer=self.optimizer,
            n_iter=20,
        )
        hyper.run()

    def score(self, X, y):
        pass
