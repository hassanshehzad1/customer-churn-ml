"""Data validation functions to ensure data quality."""

import pandas as pd
from src.config import TARGET_COLUMN, ID_COLUMN


def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Validate raw data and report quality issues.

    This function performs critical validation checks on the raw telco churn dataset.
    Critical checks will raise AssertionError if they fail. Non-critical issues are
    reported in the returned dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to validate.

    Returns
    -------
    dict
        A dictionary containing validation results with the following keys:
        - shape: tuple of (rows, columns)
        - duplicates: number of duplicate rows based on ID_COLUMN
        - missing_per_column: dict of column names to missing value counts
        - unexpected_churn_values: list of unexpected values in TARGET_COLUMN
        - blank_totalcharges_count: number of blank/whitespace-only values in TotalCharges

    Raises
    ------
    AssertionError
        If TARGET_COLUMN or ID_COLUMN are not present in the DataFrame columns.
    """
    # Critical checks - these will fail loudly if not met
    assert TARGET_COLUMN in df.columns, f"Critical: TARGET_COLUMN '{TARGET_COLUMN}' not found in DataFrame columns"
    assert ID_COLUMN in df.columns, f"Critical: ID_COLUMN '{ID_COLUMN}' not found in DataFrame columns"

    # Count duplicate rows based on ID_COLUMN
    duplicate_count = df[ID_COLUMN].duplicated().sum()

    # Count missing values per column
    missing_per_column = df.isnull().sum().to_dict()

    # Check Churn column for unexpected values
    expected_churn_values = {"Yes", "No"}
    actual_churn_values = set(df[TARGET_COLUMN].dropna().unique())
    unexpected_churn_values = list(actual_churn_values - expected_churn_values)

    # Check TotalCharges column for blank/whitespace-only values
    blank_totalcharges_count = 0
    if "TotalCharges" in df.columns:
        blank_totalcharges_count = df["TotalCharges"].astype(str).str.strip().eq("").sum()

    # Compile validation summary
    validation_summary = {
        "shape": df.shape,
        "duplicates": duplicate_count,
        "missing_per_column": missing_per_column,
        "unexpected_churn_values": unexpected_churn_values,
        "blank_totalcharges_count": blank_totalcharges_count
    }

    return validation_summary
