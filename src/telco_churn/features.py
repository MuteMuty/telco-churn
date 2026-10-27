"""Feature engineering and preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from telco_churn.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Numeric columns created by add_features (they get scaled like the original numeric ones)
ENGINEERED_COLUMNS = [
    "AvgMonthlyCharges",
    "NumAddons",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with engineered feature columns added."""
    X = X.copy()

    # TotalCharges / tenure; new customers (tenure == 0) have no bill history,
    # so use MonthlyCharges — their expected average is the current price.
    X["AvgMonthlyCharges"] = (X["TotalCharges"] / X["tenure"]).where(
        X["tenure"] != 0, X["MonthlyCharges"]
    )

    X["NumAddons"] = X[ADDON_COLUMNS].eq("Yes").sum(axis=1)

    return X


def _add_features_names_out(transformer, input_features) -> list[str]:
    """Column names after add_features, so the pipeline can report its feature names."""
    return [*input_features, *ENGINEERED_COLUMNS]


def build_preprocessor() -> Pipeline:
    """Raw features -> model-ready numbers: add features, scale numeric, one-hot encode categorical."""
    encode = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        [
            (
                "add_features",
                FunctionTransformer(
                    add_features, feature_names_out=_add_features_names_out
                ),
            ),
            ("encode", encode),
        ]
    )
