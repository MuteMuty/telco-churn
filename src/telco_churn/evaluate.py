"""Evaluation: decision threshold, test-set metrics and feature importance."""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from telco_churn.model import RANDOM_STATE, build_model

# Candidate decision thresholds: 0.05, 0.06, ..., 0.94
# (rounded, because np.arange gives values like 0.31000000000000005)
THRESHOLDS = np.round(np.arange(0.05, 0.95, 0.01), 2)


def out_of_fold_proba(
    classifier: ClassifierMixin, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> np.ndarray:
    """Churn probability for every customer, each predicted by a model that didn't see them.

    Used on the training set to choose the decision threshold without touching
    the test set.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(
        build_model(classifier), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )
    return proba[:, 1]  # columns are [P(stays), P(churns)]


def threshold_table(
    y_true: pd.Series, proba: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> pd.DataFrame:
    """Precision, recall, F1 and share of customers flagged, for each threshold."""
    rows = []
    for t in thresholds:
        predicted = (proba >= t).astype(int)
        rows.append(
            {
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                # share of customers flagged = who the sales team would contact
                "flagged_share": float(predicted.mean()),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(thresholds, name="threshold"))


def choose_threshold(
    y_true: pd.Series, proba: np.ndarray, min_recall: float = 0.75
) -> float:
    """Highest-precision threshold that still catches at least `min_recall` of churners."""
    table = threshold_table(y_true, proba)
    eligible = table[table["recall"] >= min_recall]
    if eligible.empty:
        raise ValueError(f"No threshold reaches a recall of {min_recall}")
    return float(eligible["precision"].idxmax())


def evaluation_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float
) -> pd.Series:
    """All test-set metrics in one place."""
    predicted = (proba >= threshold).astype(int)
    return pd.Series(
        {
            # threshold-free: how well the model ranks churners above non-churners
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
            # at the threshold: the yes/no decisions the sales team acts on
            "precision": precision_score(y_true, predicted, zero_division=0),
            "recall": recall_score(y_true, predicted, zero_division=0),
            "f1": f1_score(y_true, predicted, zero_division=0),
            # always predicting "stays" is already 73.5% accurate; only a reference
            "accuracy": accuracy_score(y_true, predicted),
        }
    )


def feature_importance(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance per original feature.

    How much PR-AUC drops when one feature's values are shuffled between customers.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {"importance": result.importances_mean, "std": result.importances_std},
        index=X.columns,
    ).sort_values("importance", ascending=False)
