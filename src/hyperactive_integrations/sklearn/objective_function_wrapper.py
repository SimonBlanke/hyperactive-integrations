# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_validate


class ObjectiveFunctionWrapper:
    def __init__(self, estimator, X, y, cv=3) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv

    def objective_function(self, params):
        cv_results = cross_validate(
            self.estimator,
            self.X,
            self.y,
            cv=self.cv,
        )

        return cv_results["test_score"].mean()
