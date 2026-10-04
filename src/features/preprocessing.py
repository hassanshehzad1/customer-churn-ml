"""Feature engineering and data preprocessing functions."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_totalcharges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalCharges column by stripping whitespace, replacing empty strings with NaN,
    and converting to numeric type.

    This function operates on a DataFrame and only modifies the TotalCharges column,
    leaving all other columns unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the TotalCharges column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned TotalCharges column.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        # Strip whitespace from TotalCharges
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        # Replace empty strings with NaN
        df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
        # Convert to numeric (float)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_numeric_pipeline() -> Pipeline:
    """
    Build a preprocessing pipeline for numeric columns.

    The pipeline performs the following steps in order:
    1. Clean TotalCharges (strip whitespace, replace empty with NaN, convert to numeric)
    2. Impute missing values with median
    3. Scale features using StandardScaler

    Returns
    -------
    Pipeline
        sklearn Pipeline for numeric preprocessing.
    """
    pipeline = Pipeline([
        ("clean_totalcharges", FunctionTransformer(clean_totalcharges)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return pipeline


def build_categorical_pipeline() -> Pipeline:
    """
    Build a preprocessing pipeline for categorical columns.

    The pipeline performs the following steps in order:
    1. Impute missing values with most frequent value
    2. One-hot encode categories

    Returns
    -------
    Pipeline
        sklearn Pipeline for categorical preprocessing.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return pipeline


def build_full_preprocessor() -> ColumnTransformer:
    """
    Build a full preprocessor that combines numeric and categorical pipelines.

    This function creates a ColumnTransformer that applies different preprocessing
    steps to different column types:
    - Numeric columns: clean, impute with median, scale
    - Categorical columns: impute with most frequent, one-hot encode

    Returns
    -------
    ColumnTransformer
        sklearn ColumnTransformer that applies appropriate preprocessing to each
        column type.
    """
    # Define column lists (hardcoded for stability and explicitness)
    NUMERIC_COLUMNS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen"
    ]

    CATEGORICAL_COLUMNS = [
        "gender",
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
        "PaymentMethod"
    ]

    # Build ColumnTransformer that routes columns to appropriate pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERIC_COLUMNS),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_COLUMNS)
        ],
        remainder="drop"  # Drop any columns not explicitly listed
    )

    return preprocessor
