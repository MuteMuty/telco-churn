"""Model experiments behind section 4.3 of the notebook.

Not used by the notebook or the saved model; kept so the reported results can be
reproduced. Everything runs on the training set only: the test set stays untouched.

Usage:  uv run python scripts/model_experiments.py   (takes about 6 minutes)
"""

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    clean_data,
    load_raw_data,
    split_features_target,
)
from telco_churn.model import RANDOM_STATE, build_model, split_train_test

METRIC = "average_precision"  # PR-AUC, as in the notebook
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
REPEATED_CV = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=RANDOM_STATE
)

# Per model family: (default classifier, search space, n_iter).
# "model__" addresses the classifier step of the pipeline.
# n_iter=None tries every combination; otherwise n_iter random combinations.
SEARCHES = {
    "Logistic regression": (
        LogisticRegression(class_weight="balanced", solver="saga", max_iter=5000),
        {"model__C": np.logspace(-3, 2, 11), "model__l1_ratio": [0.0, 0.5, 1.0]},
        None,
    ),
    "Random forest": (
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        {
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_leaf": [1, 5, 10, 20, 40],
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
        30,
    ),
    "Gradient boosting": (
        HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        {
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_iter": [100, 200, 400],
            "model__max_depth": [2, 3, 4, 6, None],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__l2_regularization": loguniform(1e-3, 10),
        },
        30,
    ),
}


def make_search(name, cv):
    """Grid or random search for one model family, scored by PR-AUC."""
    classifier, space, n_iter = SEARCHES[name]
    pipeline = build_model(clone(classifier))
    if n_iter is None:
        return GridSearchCV(pipeline, space, scoring=METRIC, cv=cv, n_jobs=-1)
    return RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring=METRIC,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def hyperparameter_search(X, y):
    """Default vs best settings per family. Returns a summary and the best classifiers."""
    rows, best = {}, {}
    for name, (classifier, _, _) in SEARCHES.items():
        default = cross_val_score(
            build_model(clone(classifier)), X, y, cv=CV, scoring=METRIC
        )
        search = make_search(name, CV).fit(X, y)
        best[name] = clone(search.best_estimator_.named_steps["model"])
        rows[name] = {
            "default_pr_auc": default.mean(),
            "tuned_pr_auc": search.best_score_,
            "best_settings": {
                # float(): print 0.3162, not np.float64(0.3162)
                key.removeprefix("model__"): round(float(value), 4)
                if isinstance(value, float)
                else value
                for key, value in search.best_params_.items()
            },
        }
    return pd.DataFrame.from_dict(rows, orient="index"), best


def nested_cv(X, y):
    """Honest score of each tuned family: the search runs inside each outer fold."""
    rows = {}
    for name in SEARCHES:
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(make_search(name, inner), X, y, cv=CV, scoring=METRIC)
        rows[name] = {"nested_pr_auc": scores.mean(), "std": scores.std()}
    return pd.DataFrame(rows).T


def ensemble_comparison(best, X, y):
    """Tuned models and a soft-voting ensemble (LR + GB), 5 x 5 repeated CV."""
    models = {
        **best,
        "Ensemble (LR + GB)": VotingClassifier(
            [("lr", best["Logistic regression"]), ("gb", best["Gradient boosting"])],
            voting="soft",
        ),
    }
    folds = pd.DataFrame(
        {
            name: cross_val_score(
                build_model(model), X, y, cv=REPEATED_CV, scoring=METRIC, n_jobs=-1
            )
            for name, model in models.items()
        }
    )
    better = folds.sub(folds["Logistic regression"], axis=0) > 0
    return pd.DataFrame(
        {
            "pr_auc": folds.mean(),
            "std": folds.std(),
            "folds_better_than_LR": better.sum(),
        }
    )


def class_weight_comparison(best, X, y):
    """Ranking quality and probability quality with and without class weights."""
    rows = {}
    for name in ["Logistic regression", "Gradient boosting"]:
        for class_weight in ["balanced", None]:
            classifier = clone(best[name]).set_params(class_weight=class_weight)
            proba = cross_val_predict(
                build_model(classifier), X, y, cv=CV, method="predict_proba", n_jobs=-1
            )[:, 1]
            rows[(name, str(class_weight))] = {
                "pr_auc": average_precision_score(y, proba),
                "mean_predicted_proba": proba.mean(),
                "actual_churn_rate": y.mean(),
                "brier_score": brier_score_loss(y, proba),
            }
    return pd.DataFrame(rows).T


def engineered_features_comparison(best, X, y):
    """PR-AUC with and without the engineered features, 5 x 5 repeated CV."""
    # same preprocessing as build_preprocessor, minus the engineered features
    without_engineered = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    rows = {}
    for name in ["Logistic regression", "Gradient boosting"]:
        with_pipeline = build_model(clone(best[name]))
        without_pipeline = Pipeline(
            [("preprocess", without_engineered), ("model", clone(best[name]))]
        )
        rows[name] = {
            label: cross_validate(
                pipeline, X, y, cv=REPEATED_CV, scoring=METRIC, n_jobs=-1
            )["test_score"].mean()
            for label, pipeline in [
                ("with_engineered", with_pipeline),
                ("without", without_pipeline),
            ]
        }
    return pd.DataFrame(rows).T


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    X, y = split_features_target(clean_data(load_raw_data()))
    X_train, _, y_train, _ = split_train_test(X, y)

    print("1. Hyperparameter search (5-fold CV PR-AUC)")
    search_results, best = hyperparameter_search(X_train, y_train)
    print(search_results.round(3), "\n")

    print("2. Nested CV (honest PR-AUC of each tuned family)")
    print(nested_cv(X_train, y_train).round(3), "\n")

    print("3. Ensemble vs tuned models (5 x 5 repeated CV)")
    print(ensemble_comparison(best, X_train, y_train).round(3), "\n")

    print("4. Class weights")
    print(class_weight_comparison(best, X_train, y_train).round(3), "\n")

    print("5. Engineered features (5 x 5 repeated CV PR-AUC)")
    print(engineered_features_comparison(best, X_train, y_train).round(4))


if __name__ == "__main__":
    main()
