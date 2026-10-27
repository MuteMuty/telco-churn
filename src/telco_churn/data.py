"""Loading and cleaning the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as provided."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    - TotalCharges: text -> float; blanks (tenure == 0, never billed) -> 0
    - SeniorCitizen: 0/1 -> "No"/"Yes", consistent with the other binary columns
    - Churn: "Yes"/"No" -> 1/0
    - drop customerID (identifier, not a feature)
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    df = df.drop(columns=[ID_COLUMN])

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features X and target y."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    return X, y
