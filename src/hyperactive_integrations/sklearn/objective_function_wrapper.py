# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_validate


class ObjectiveFunctionWrapper:
    def __init__(self, estimator, X, y) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y

    def objective_function(self, params):
        scores = cross_validate(
            self.estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score=self.error_score,
            params=self.fit_params,
            groups=self.groups,
            return_train_score=self.return_train_score,
            scoring=self.scoring,
        )

        return scores["test_score"].mean()
