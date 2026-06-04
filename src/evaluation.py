"""Evaluation utilities.

Two families of metrics are reported:

1. Generic classification metrics (accuracy, F1, ROC-AUC, confusion matrix) per
   horizon, the usual lens for standard classifiers.
2. The competition metrics, so the classification track is directly comparable
   with the teammate's survival track:

       Hybrid = 0.3 * C-index + 0.7 * (1 - Weighted Brier)
       Weighted Brier = 0.3 * Brier_24h + 0.4 * Brier_48h + 0.3 * Brier_72h

   For each Brier horizon, rows censored before the horizon are excluded and rows
   censored after it count as no-event (handled by ``horizon_labels``).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .data_loader import EVENT_COL, TIME_COL, horizon_labels, make_horizon_dataset
from .utils import HORIZONS, RANDOM_STATE, enforce_monotonic

BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}


# --------------------------------------------------------------------------- #
# Generic per-horizon classification CV
# --------------------------------------------------------------------------- #
def cv_classification(pipe, X, y, n_splits=5, random_state=RANDOM_STATE):
    """Stratified k-fold metrics plus out-of-fold probabilities for one horizon."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "roc_auc"]
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    oof_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return {
        "accuracy": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),
        "roc_auc": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "oof_prob": oof_prob,
        "y_true": y.to_numpy(),
    }


def evaluate_models_per_horizon(models, preprocessor_factory, train, features,
                                horizons=HORIZONS, n_splits=5, random_state=RANDOM_STATE):
    """Run CV for every (model, horizon) pair.

    ``preprocessor_factory()`` must return a fresh unfitted preprocessing step.
    Returns a tidy summary dataframe and a nested dict of OOF arrays.
    """
    from sklearn.pipeline import Pipeline

    rows, oof = [], {}
    for name, cfg in models.items():
        oof[name] = {}
        for h in horizons:
            X, y = make_horizon_dataset(train, h, features)
            pipe = Pipeline([("pre", preprocessor_factory()), ("model", cfg["estimator"])])
            res = cv_classification(pipe, X, y, n_splits=n_splits, random_state=random_state)
            oof[name][h] = res
            rows.append({
                "model": name, "horizon_h": h,
                "accuracy": res["accuracy"], "f1": res["f1"], "roc_auc": res["roc_auc"],
                "roc_auc_std": res["roc_auc_std"],
            })
    return pd.DataFrame(rows), oof


# --------------------------------------------------------------------------- #
# Competition metrics
# --------------------------------------------------------------------------- #
def concordance_index(time, event, risk):
    """Harrell's C-index. Higher ``risk`` should mean an earlier event."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)
    num = den = 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den if den > 0 else 0.5


def weighted_brier(train, prob_by_horizon):
    """Weighted Brier score over horizons 24/48/72 with the censoring rule.

    ``prob_by_horizon`` maps horizon -> predicted P(hit by H) for ALL train rows.
    """
    total = 0.0
    parts = {}
    for h, w in BRIER_WEIGHTS.items():
        usable, y = horizon_labels(train, h)
        p = np.asarray(prob_by_horizon[h])[usable]
        b = float(np.mean((p - y[usable]) ** 2))
        parts[h] = b
        total += w * b
    return total, parts


def hybrid_score(train, prob_matrix, horizons=HORIZONS, risk=None):
    """Full competition score from an OOF probability matrix [n_rows x n_horizons]."""
    prob_by_h = {h: prob_matrix[:, i] for i, h in enumerate(horizons)}
    if risk is None:
        risk = prob_matrix.mean(axis=1)  # more early probability mass => higher urgency
    c = concordance_index(train[TIME_COL].to_numpy(), train[EVENT_COL].to_numpy(), risk)
    wb, parts = weighted_brier(train, prob_by_h)
    score = 0.3 * c + 0.7 * (1.0 - wb)
    return {"hybrid": score, "c_index": c, "weighted_brier": wb, "brier_parts": parts}


def competition_cv(estimator_factory, preprocessor_factory, train, features,
                   horizons=HORIZONS, n_splits=5, random_state=RANDOM_STATE,
                   monotonic=True):
    """Competition Hybrid Score for a single model via shared stratified CV.

    ``estimator_factory(h)`` returns a fresh classifier for horizon h; ``preprocessor_factory()``
    a fresh preprocessing step. Returns the score dict and the OOF prob matrix.
    """
    from sklearn.pipeline import Pipeline

    n = len(train)
    prob = np.full((n, len(horizons)), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strat = train[EVENT_COL].to_numpy()

    for tr_idx, te_idx in skf.split(np.arange(n), strat):
        tr_set = train.iloc[tr_idx].reset_index(drop=True)
        X_te = train.iloc[te_idx][features]
        for j, h in enumerate(horizons):
            usable, y = horizon_labels(tr_set, h)
            X_tr = tr_set.loc[usable, features]
            y_tr = y[usable]
            if len(np.unique(y_tr)) < 2:
                prob[te_idx, j] = float(np.mean(y_tr)) if len(y_tr) else 0.5
                continue
            pipe = Pipeline([("pre", preprocessor_factory()), ("model", estimator_factory(h))])
            pipe.fit(X_tr, y_tr)
            prob[te_idx, j] = pipe.predict_proba(X_te)[:, 1]

    if monotonic:
        prob = enforce_monotonic(prob)
    return hybrid_score(train, prob, horizons=horizons), prob


# --------------------------------------------------------------------------- #
# Final submission
# --------------------------------------------------------------------------- #
def fit_horizon_models(estimator_factory, preprocessor_factory, train, features,
                       horizons=HORIZONS):
    """Fit one classifier per horizon on the full usable training data."""
    from sklearn.pipeline import Pipeline

    fitted = {}
    for h in horizons:
        X, y = make_horizon_dataset(train, h, features)
        pipe = Pipeline([("pre", preprocessor_factory()), ("model", estimator_factory(h))])
        pipe.fit(X, y)
        fitted[h] = pipe
    return fitted


def make_submission(fitted, X_test, ids, horizons=HORIZONS, monotonic=True):
    """Predict the four horizon probabilities and assemble the submission frame."""
    prob = np.column_stack([fitted[h].predict_proba(X_test)[:, 1] for h in horizons])
    if monotonic:
        prob = enforce_monotonic(prob)
    out = pd.DataFrame({"event_id": np.asarray(ids)})
    for j, h in enumerate(horizons):
        out[f"prob_{h}h"] = prob[:, j]
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_model_comparison(summary, metric="roc_auc"):
    pivot = summary.pivot(index="model", columns="horizon_h", values=metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison by horizon ({metric})")
    ax.legend(title="horizon (h)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_roc_curves(oof_for_model, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    for h, res in oof_for_model.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["oof_prob"])
        auc = roc_auc_score(res["y_true"], res["oof_prob"])
        ax.plot(fpr, tpr, label=f"{h}h (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves - {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrices(oof_for_model, model_name, threshold=0.5):
    horizons = list(oof_for_model.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.2 * len(horizons), 3))
    for ax, h in zip(np.atleast_1d(axes), horizons):
        res = oof_for_model[h]
        pred = (res["oof_prob"] >= threshold).astype(int)
        cm = confusion_matrix(res["y_true"], pred)
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center")
        ax.set_title(f"{h}h")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    fig.suptitle(f"Confusion matrices - {model_name}")
    fig.tight_layout()
    return fig


def plot_feature_importance(fitted_pipe, feature_names, top=15, model_name=""):
    model = fitted_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    imp = imp.tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature importance - {model_name}")
    fig.tight_layout()
    return fig
