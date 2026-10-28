"""Candidate models, cross-validation, training and saving."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.data import PROJECT_ROOT
from telco_churn.features import build_preprocessor

RANDOM_STATE = 42
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Metrics computed in every CV fold, all for the churn class (1). Step 5 explains them.
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a stratified test set, used only for the final evaluation (step 5)."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def build_model(classifier: ClassifierMixin) -> Pipeline:
    """Preprocessing + classifier in one pipeline: raw customer data in, prediction out."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", classifier)])


def candidate_models() -> dict[str, ClassifierMixin]:
    """The models compared in the notebook, by display name.

    Settings are the best found by hyperparameter searches during development
    (notebook, section 4.2). The default random forest is included to show
    what overfitting looks like.
    """
    return {
        # C: regularisation strength; l1_ratio=1.0: L1 penalty, can switch columns off
        "Logistic regression": LogisticRegression(
            C=0.3, l1_ratio=1.0, solver="saga", max_iter=5000
        ),
        "Random forest (default settings)": RandomForestClassifier(
            random_state=RANDOM_STATE
        ),
        # every leaf must hold at least 20 customers, so trees can't memorise
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.3,
            random_state=RANDOM_STATE,
        ),
        # many small (depth-2) trees, each correcting the previous ones a little
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=2, max_iter=200, random_state=RANDOM_STATE
        ),
    }


def cross_validate_models(
    models: dict[str, ClassifierMixin],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Stratified k-fold CV of each model; one row per model with mean scores."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, classifier in models.items():
        scores = cross_validate(
            build_model(classifier),
            X,
            y,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=-1,
        )
        results[name] = {
            key: values.mean()
            for key, values in scores.items()
            if key.startswith(("test_", "train_"))
        }
        results[name]["test_pr_auc_std"] = scores["test_pr_auc"].std()
    return pd.DataFrame(results).T


def train_model(classifier: ClassifierMixin, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the full pipeline for `classifier` on X, y."""
    return build_model(classifier).fit(X, y)


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save a fitted pipeline to disk."""
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline saved with save_model."""
    return joblib.load(path)
