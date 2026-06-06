"""Model definitions and hyperparameter grids.

The classifiers and tuning ranges for Logistic Regression, Random Forest and
Gradient Boosting follow the preliminary documentation (Table 1 and Table 2).
Decision Tree, SVM and k-NN are added as additional standard-classifier
baselines to broaden the comparison; this extension is noted in the README.
Grids are expressed for use inside a Pipeline (prefix ``model__``).
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.util import Surv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .utils import RANDOM_STATE


class KaplanMeierBaseline:
    """Feature-independent Kaplan-Meier baseline."""
    def fit(self, X, time, event):
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        event_times = np.sort(np.unique(time[event]))

        surv = []
        current = 1.0
        for t in event_times:
            at_risk = np.sum(time >= t)
            observed = np.sum((time == t) & event)
            if at_risk > 0:
                current *= 1.0 - observed / at_risk
            surv.append(current)

        self.event_times_ = event_times
        self.survival_ = np.asarray(surv, dtype=float)
        return self

    def _survival_at(self, horizons):
        horizons = np.asarray(horizons, dtype=float)
        idx = np.searchsorted(self.event_times_, horizons, side="right") - 1
        out = np.ones_like(horizons, dtype=float)
        valid = idx >= 0
        out[valid] = self.survival_[idx[valid]]
        return out

    def predict_event_probability(self, X, horizons):
        p = 1.0 - self._survival_at(horizons)
        return np.tile(p, (len(X), 1))


class SurvivalAdapter:
    """Small adapter exposing ``predict_event_probability`` for survival models."""
    def __init__(self, estimator, backend):
        self.estimator = estimator
        self.backend = backend

    def fit(self, X, time, event):
        if self.backend == "lifelines":
            self.feature_names_ = list(getattr(X, "columns", [f"x{i}" for i in range(X.shape[1])]))
            df = pd.DataFrame(X, columns=self.feature_names_).copy()
            df["time_to_hit_hours"] = np.asarray(time, dtype=float)
            df["event"] = np.asarray(event, dtype=int)
            self.estimator.fit(df, duration_col="time_to_hit_hours", event_col="event")
        elif self.backend == "sksurv":
            y = Surv.from_arrays(np.asarray(event, dtype=bool), np.asarray(time, dtype=float))
            self.estimator.fit(X, y)
        else:
            raise ValueError(f"Unknown survival backend: {self.backend}")
        return self

    def predict_event_probability(self, X, horizons):
        if self.backend == "lifelines":
            df = pd.DataFrame(X, columns=self.feature_names_)
            sf = self.estimator.predict_survival_function(df, times=list(horizons))
            return np.clip(1.0 - sf.T.to_numpy(), 0.0, 1.0)

        funcs = self.estimator.predict_survival_function(X)
        prob = np.zeros((len(funcs), len(horizons)), dtype=float)
        for i, fn in enumerate(funcs):
            values = []
            for h in horizons:
                try:
                    values.append(float(fn(h)))
                except ValueError:
                    # scikit-survival step functions are bounded by observed
                    # train times in the fold. Past the last step, survival
                    # remains at the final value.
                    if hasattr(fn, "x") and hasattr(fn, "y"):
                        values.append(float(fn.y[0] if h < fn.x[0] else fn.y[-1]))
                    else:
                        raise
            surv = np.asarray(values, dtype=float)
            prob[i, :] = 1.0 - surv
        return np.clip(prob, 0.0, 1.0)


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


def get_default_models(random_state: int = RANDOM_STATE, fast: bool = False) -> dict:
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


def get_survival_models(random_state: int = RANDOM_STATE, fast: bool = False) -> dict:
    """Return survival model factories and compact tuning grids.
    """
    return {
        "kaplan_meier": {"factory": lambda **_: KaplanMeierBaseline(), "param_grid": {}},
        "cox_ph": {
            "factory": lambda **params: SurvivalAdapter(
                CoxPHFitter(**{"penalizer": 0.1, **params}), "lifelines"
            ),
            "param_grid": {"penalizer": [0.1, 1.0] if fast else [0.1, 0.5, 1.0]},
        },
        "random_survival_forest": {
            "factory": lambda **params: SurvivalAdapter(
                RandomSurvivalForest(random_state=random_state, n_jobs=-1, **params),
                "sksurv",
            ),
            "param_grid": {
                "n_estimators": [100, 300] if fast else [100, 300, 500],
                "min_samples_leaf": [5, 10] if fast else [5, 10, 20],
            },
        },
        "gradient_boosting_survival": {
            "factory": lambda **params: SurvivalAdapter(
                GradientBoostingSurvivalAnalysis(random_state=random_state, **params),
                "sksurv",
            ),
            "param_grid": {
                "learning_rate": [0.05, 0.1] if fast else [0.01, 0.05, 0.1],
                "max_depth": [2, 3] if fast else [2, 3, 4],
            },
        },
    }
