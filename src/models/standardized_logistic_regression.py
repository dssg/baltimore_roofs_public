from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class StandardizedLogisticRegression(Pipeline):
    # Because sklearn is strict about parameters, we need to get all of these
    # separately. See https://scikit-learn.org/stable/developers/develop.html
    # Everything is just copied from
    # https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/linear_model/_logistic.py#L1026
    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        steps = [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    penalty=self.penalty,
                    dual=self.dual,
                    tol=self.tol,
                    C=self.C,
                    fit_intercept=self.fit_intercept,
                    intercept_scaling=self.intercept_scaling,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    multi_class=self.multi_class,
                    verbose=self.verbose,
                    warm_start=self.warm_start,
                    n_jobs=self.n_jobs,
                    l1_ratio=self.l1_ratio,
                ),
            ),
        ]
        super().__init__(steps)
