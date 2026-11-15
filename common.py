"""Shared data loading and feature definitions used by the notebook, train.py and predict.py."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "data.csv"
TARGET = "Target"
RANDOM_STATE = 42

# Columns that hold *codes* (nominal categories), not quantities. Their numbers have no order,
# so they are one-hot encoded. Names are the cleaned (snake_case) names.
CATEGORICAL_FEATURES = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]


def clean_column_name(name: str) -> str:
    """'Mother's qualification' -> 'mothers_qualification', 'Daytime/evening attendance\\t' -> 'daytime_evening_attendance'."""
    name = name.strip().lower().replace("'", "")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return "nationality" if name == "nacionality" else name  # the source file misspells it


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the UCI file (semicolon separated) and clean the column names. The target keeps the name 'target'."""
    df = pd.read_csv(path, sep=";")
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical, numerical) feature names, excluding the target."""
    features = [c for c in df.columns if c != TARGET.lower()]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    numerical = [c for c in features if c not in categorical]
    return categorical, numerical


def make_preprocessor(categorical: list[str], numerical: list[str], scale: bool = False) -> ColumnTransformer:
    """One-hot encode the code columns (rare codes are grouped, unseen codes are ignored).
    Numeric columns are passed through, or standardized when `scale=True` (needed for linear models)."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
                categorical,
            ),
            ("num", num_step, numerical),
        ]
    )
