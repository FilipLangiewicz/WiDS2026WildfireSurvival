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
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

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


def evaluate_models_per_horizon(
    models,
    preprocessor_factory,
    train,
    features,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """Run CV for every (model, horizon) pair.

    ``preprocessor_factory()`` must return a fresh unfitted preprocessing step.
    Returns a tidy summary dataframe and a nested dict of OOF arrays.
    """
    rows, oof = [], {}
    for name, cfg in models.items():
        oof[name] = {}
        for h in horizons:
            X, y = make_horizon_dataset(train, h, features)
            pipe = Pipeline([("pre", preprocessor_factory()), ("model", cfg["estimator"])])
            res = cv_classification(pipe, X, y, n_splits=n_splits, random_state=random_state)
            oof[name][h] = res
            rows.append(
                {
                    "model": name,
                    "horizon_h": h,
                    "accuracy": res["accuracy"],
                    "f1": res["f1"],
                    "roc_auc": res["roc_auc"],
                    "roc_auc_std": res["roc_auc_std"],
                }
            )
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


def competition_cv(
    estimator_factory,
    preprocessor_factory,
    train,
    features,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
    monotonic=True,
):
    """Competition Hybrid Score for a single model via shared stratified CV.

    ``estimator_factory(h)`` returns a fresh classifier for horizon h; ``preprocessor_factory()``
    a fresh preprocessing step. Returns the score dict and the OOF prob matrix.
    """
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


def _survival_cv_prob(
    model_factory,
    params,
    preprocessor_factory,
    train,
    features,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
    monotonic=True,
):
    n = len(train)
    prob = np.full((n, len(horizons)), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strat = train[EVENT_COL].to_numpy()

    for tr_idx, te_idx in skf.split(np.arange(n), strat):
        tr_set = train.iloc[tr_idx]
        te_set = train.iloc[te_idx]

        pre = preprocessor_factory()
        X_tr = pre.fit_transform(tr_set[features])
        X_te = pre.transform(te_set[features])

        model = model_factory(**params)
        model.fit(X_tr, tr_set[TIME_COL].to_numpy(), tr_set[EVENT_COL].to_numpy())
        prob[te_idx, :] = model.predict_event_probability(X_te, horizons)

    if monotonic:
        prob = enforce_monotonic(prob)
    return prob


def _best_survival_cv(
    cfg,
    train,
    features,
    preprocessor_factory,
    tune=False,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
    monotonic=True,
):
    best = None
    param_grid = cfg.get("param_grid", {}) if tune else {}

    for params in ParameterGrid(param_grid):
        prob = _survival_cv_prob(
            cfg["factory"],
            params,
            preprocessor_factory,
            train,
            features,
            horizons=horizons,
            n_splits=n_splits,
            random_state=random_state,
            monotonic=monotonic,
        )
        score = hybrid_score(train, prob, horizons=horizons)

        candidate = {"params": params, "score": score, "prob": prob}
        if best is None or score["hybrid"] > best["score"]["hybrid"]:
            best = candidate

    score = dict(best["score"])
    score["best_params"] = best["params"]
    return score, best["prob"]


def _survival_summary_rows(train, model_name, prob_matrix, score, horizons=HORIZONS):
    metric_fields = {
        "hybrid": score["hybrid"],
        "c_index": score["c_index"],
        "weighted_brier": score["weighted_brier"],
        "brier_24h": score["brier_parts"][24],
        "brier_48h": score["brier_parts"][48],
        "brier_72h": score["brier_parts"][72],
        "best_params": str(score["best_params"]),
    }

    td_auc, td_times, mean_td_auc = _time_dependent_auc(train, prob_matrix, horizons=horizons)
    rows = []
    for j, h in enumerate(horizons):
        usable, y = horizon_labels(train, h)
        y_h = y[usable]
        p_h = prob_matrix[usable, j]
        roc = roc_auc_score(y_h, p_h) if len(np.unique(y_h)) == 2 else np.nan
        brier = float(np.mean((p_h - y_h) ** 2))
        rows.append(
            {
                "model": model_name,
                "horizon_h": h,
                "roc_auc": roc,
                "brier": brier,
                "td_auc": td_auc[h],
                "td_auc_time_h": td_times[h],
                "mean_td_auc": mean_td_auc,
                "usable": int(usable.sum()),
                "positives": int(y_h.sum()),
                **metric_fields,
            }
        )
    return rows


def evaluate_survival_models(
    survival_models,
    preprocessor_factory,
    train,
    features,
    model_names=None,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
    monotonic=True,
    tune=False,
):
    """Evaluate survival models, optionally tuning over each model's ``param_grid``."""
    rows = []
    best_prob = {}
    names = model_names or survival_models.keys()

    for name in names:
        score, prob = _best_survival_cv(
            survival_models[name],
            train,
            features,
            preprocessor_factory,
            tune=tune,
            horizons=horizons,
            n_splits=n_splits,
            random_state=random_state,
            monotonic=monotonic,
        )
        rows.extend(_survival_summary_rows(train, name, prob, score, horizons=horizons))
        if prob is not None:
            best_prob[name] = prob

    summary = pd.DataFrame(rows).sort_values(
        ["hybrid", "model", "horizon_h"],
        ascending=[False, True, True],
        na_position="last",
    )
    return summary.reset_index(drop=True), best_prob


def _time_dependent_auc(train, prob_matrix, horizons=HORIZONS):
    time = train[TIME_COL].to_numpy(dtype=float)
    event = train[EVENT_COL].to_numpy(dtype=bool)
    y_surv = Surv.from_arrays(event=event, time=time)

    max_time = np.max(time)
    eval_times = np.asarray([min(float(h), max_time - 1e-6) for h in horizons], dtype=float)
    auc, mean_auc = cumulative_dynamic_auc(y_surv, y_surv, prob_matrix, eval_times)

    return dict(zip(horizons, auc)), dict(zip(horizons, eval_times)), float(mean_auc)


def tuned_classification_horizon_metrics(
    tuned_best,
    preprocessor_factory,
    train,
    features,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
    suffix=" (tuned)",
):
    """Per-horizon OOF ROC-AUC for already selected per-horizon classifiers."""
    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for name, per_horizon_models in tuned_best.items():
        for h in horizons:
            X, y = make_horizon_dataset(train, h, features)
            pipe = Pipeline(
                [
                    ("pre", preprocessor_factory()),
                    ("model", clone(per_horizon_models[h])),
                ]
            )
            prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            rows.append(
                {
                    "model": f"{name}{suffix}",
                    "horizon_h": h,
                    "roc_auc": roc_auc_score(y, prob),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Final submission
# --------------------------------------------------------------------------- #
def fit_horizon_models(estimator_factory, preprocessor_factory, train, features, horizons=HORIZONS):
    """Fit one classifier per horizon on the full usable training data."""
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tuned_roc_curves_by_horizon(
    classification_tuned_best,
    preprocessor_factory,
    train,
    features,
    survival_tuned_oof=None,
    horizons=HORIZONS,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """ROC curves per horizon for tuned classifiers and tuned survival models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = np.asarray(axes).ravel()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for ax, h in zip(axes, horizons):
        X, y = make_horizon_dataset(train, h, features)

        for name, per_horizon_models in classification_tuned_best.items():
            pipe = Pipeline(
                [
                    ("pre", preprocessor_factory()),
                    ("model", clone(per_horizon_models[h])),
                ]
            )
            prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            fpr, tpr, _ = roc_curve(y, prob)
            auc = roc_auc_score(y, prob)
            ax.plot(fpr, tpr, lw=1.8, label=f"{name} cls (AUC={auc:.3f})")

        if survival_tuned_oof:
            usable, y_all = horizon_labels(train, h)
            y_h = y_all[usable]
            h_idx = list(horizons).index(h)
            for name, prob_matrix in survival_tuned_oof.items():
                prob_h = np.asarray(prob_matrix, dtype=float)[usable, h_idx]
                if len(np.unique(y_h)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_h, prob_h)
                auc = roc_auc_score(y_h, prob_h)
                ax.plot(fpr, tpr, lw=1.8, linestyle="--", label=f"{name} surv (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"Horizon {h}h")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Tuned ROC curves by horizon", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
