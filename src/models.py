"""Model definitions and hyperparameter grids.

The classifiers and tuning ranges for Logistic Regression, Random Forest and
Gradient Boosting follow the preliminary documentation (Table 1 and Table 2).
Decision Tree, SVM and k-NN are added as additional standard-classifier
baselines to broaden the comparison; this extension is noted in the README.
Grids are expressed for use inside a Pipeline (prefix ``model__``).
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from .utils import RANDOM_STATE


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """LinearRegression wrapped to expose predict_proba via clipping."""
    def __init__(self):
        self.model = LinearRegression()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        raw = self.model.predict(X)
        raw = np.where(np.isfinite(raw), raw, 0.5)
        p = np.clip(raw, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.model.predict(X) >= 0.5).astype(int)


def get_models(random_state: int = RANDOM_STATE, fast: bool = False) -> dict:
    """Return ``name -> {estimator, param_grid}``.

    When ``fast`` is True the grids are trimmed for the quick notebook variant.
    """
    models = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000, solver="liblinear", random_state=random_state
            ),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_leaf": [1, 5, 10],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [3, 5, 10, None],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "model__n_neighbors": [3, 5, 7, 11, 15],
                "model__weights": ["uniform", "distance"],
            },
        },
        "linear_regression": {
            "estimator": LinearRegressionClassifier(),
            "param_grid": {},
        },
    }

    if fast:
        fast_grids = {
            "logistic_regression": {"model__C": [0.1, 1], "model__penalty": ["l2"]},
            "decision_tree": {"model__max_depth": [3, 5], "model__min_samples_leaf": [5]},
            "random_forest": {"model__n_estimators": [100], "model__max_depth": [5, None]},
            "gradient_boosting": {"model__learning_rate": [0.05], "model__max_depth": [2, 3]},
            "svm": {"model__C": [1], "model__kernel": ["rbf"]},
            "knn": {"model__n_neighbors": [5, 11]},
            "linear_regression": {},
        }
        for name, grid in fast_grids.items():
            models[name]["param_grid"] = grid

    return models
