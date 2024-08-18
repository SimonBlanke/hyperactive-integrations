# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import BaseSearchCV
from hyperactive import Hyperactive

from .objective_function_adapter import ObjectiveFunctionAdapter


class HyperactiveSearchCV(BaseEstimator):
    def __init__(
        self,
        estimator,
        optimizer,
        params_config,
        n_iter=100,
        *,
        scoring=None,
        n_jobs=1,
        refit=True,
        cv=None,
    ):

        self.estimator = estimator
        self.optimizer = optimizer
        self.params_config = params_config
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv

    def fit(self, X, y):
        objective_function_adapter = ObjectiveFunctionAdapter(
            self.estimator,
        )
        objective_function_adapter.add_dataset(X, y)
        objective_function_adapter.add_validation(self.scoring, self.cv)

        hyper = Hyperactive()
        hyper.add_search(
            objective_function_adapter.objective_function,
            search_space=self.params_config,
            optimizer=self.optimizer,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
        )
        hyper.run()

        return self
