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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features based on Phase 3 EDA findings.

    This function adds domain-specific features that capture business insights:
    - is_new_customer: Newer customers (tenure < 6 months) have higher churn risk
    - high_risk_contract: Month-to-month customers with low tenure are at highest risk
    - avg_monthly_spend_ratio: Captures if current charges are rising relative to historical average
    - num_services: Count of services subscribed (service engagement level)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw customer data. Must contain required columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 4 new engineered features added.
    """
    df = df.copy()

    # Clean TotalCharges first (handles dirty string values)
    df = clean_totalcharges(df)

    # is_new_customer: 1 if tenure < 6 months, else 0
    # Business reasoning: Newer customers haven't established loyalty and are more likely to churn
    df["is_new_customer"] = (df["tenure"] < 6).astype(int)

    # high_risk_contract: 1 if month-to-month AND tenure < 12 months, else 0
    # Business reasoning: Month-to-month customers with short tenure are the highest risk segment
    df["high_risk_contract"] = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] < 12)
    ).astype(int)

    # avg_monthly_spend_ratio: MonthlyCharges / (TotalCharges / (tenure + 1) + 0.01)
    # Business reasoning: If current monthly charges are higher than historical average,
    # customer may be dissatisfied with price increases
    # Add 1 to tenure to avoid division by zero for new customers
    # Add 0.01 to denominator to avoid division by zero
    df["avg_monthly_spend_ratio"] = df["MonthlyCharges"] / (
        df["TotalCharges"] / (df["tenure"] + 1) + 0.01
    )

    # num_services: Count of subscribed services
    # Business reasoning: Customers with more services are more engaged and less likely to churn
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PhoneService"
    ]
    df["num_services"] = (
        df[service_columns].apply(lambda x: (x == "Yes").sum(), axis=1) +
        (df["InternetService"] != "No").astype(int)
    )

    return df


def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer only non-redundant features based on Phase 3 EDA findings.

    This function adds only 2 domain-specific features that are not redundant
    with existing columns:
    - avg_monthly_spend_ratio: Captures if current charges are rising relative to historical average
    - num_services: Count of services subscribed (service engagement level)

    Excludes is_new_customer and high_risk_contract as they are redundant with
    existing tenure and Contract columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw customer data. Must contain required columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 2 new engineered features added.
    """
    df = df.copy()

    # Clean TotalCharges first (handles dirty string values)
    df = clean_totalcharges(df)

    # avg_monthly_spend_ratio: MonthlyCharges / (TotalCharges / (tenure + 1) + 0.01)
    # Business reasoning: If current monthly charges are higher than historical average,
    # customer may be dissatisfied with price increases
    # Add 1 to tenure to avoid division by zero for new customers
    # Add 0.01 to denominator to avoid division by zero
    df["avg_monthly_spend_ratio"] = df["MonthlyCharges"] / (
        df["TotalCharges"] / (df["tenure"] + 1) + 0.01
    )

    # num_services: Count of subscribed services
    # Business reasoning: Customers with more services are more engaged and less likely to churn
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PhoneService"
    ]
    df["num_services"] = (
        df[service_columns].apply(lambda x: (x == "Yes").sum(), axis=1) +
        (df["InternetService"] != "No").astype(int)
    )

    return df


def build_numeric_pipeline() -> Pipeline:
    """
    Build a preprocessing pipeline for numeric columns.

    The pipeline performs the following steps in order:
    1. Impute missing values with median
    2. Scale features using StandardScaler

    Note: TotalCharges cleaning is now done in engineer_features() before this pipeline.

    Returns
    -------
    Pipeline
        sklearn Pipeline for numeric preprocessing.
    """
    pipeline = Pipeline([
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


def build_full_preprocessor() -> Pipeline:
    """
    Build a full preprocessor with NO feature engineering (original Phase 4/5 behavior).

    This function creates a Pipeline that:
    1. Cleans TotalCharges column (strips whitespace, converts to numeric)
    2. Applies ColumnTransformer for numeric and categorical preprocessing:
       - Numeric columns: impute with median, scale
       - Categorical columns: impute with most frequent, one-hot encode

    This is the original preprocessor used in Phase 5 baseline, which achieved
    better Recall and F1 than the version with engineered features.

    Returns
    -------
    Pipeline
        sklearn Pipeline with TotalCharges cleaning + column transformations.
    """
    # Define column lists (original 4 numeric + 15 categorical, no engineered features)
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
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERIC_COLUMNS),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_COLUMNS)
        ],
        remainder="drop"  # Drop any columns not explicitly listed
    )

    # Build outer Pipeline: clean TotalCharges first, then column transformations
    preprocessor = Pipeline([
        ("clean_totalcharges", FunctionTransformer(clean_totalcharges)),
        ("column_transformer", column_transformer)
    ])

    return preprocessor


def build_full_preprocessor_with_engineered_features() -> Pipeline:
    """
    Build a full preprocessor that applies feature engineering then column transformations.

    This function creates a Pipeline that:
    1. Applies feature engineering (adds 4 new engineered features)
    2. Applies ColumnTransformer for numeric and categorical preprocessing:
       - Numeric columns: clean, impute with median, scale
       - Categorical columns: impute with most frequent, one-hot encode

    This version includes engineered features (is_new_customer, high_risk_contract,
    avg_monthly_spend_ratio, num_services) which were tested in Phase 6 but found
    to have worse Recall/F1 than the baseline.

    Returns
    -------
    Pipeline
        sklearn Pipeline that applies feature engineering then column transformations.
    """
    # Define column lists (hardcoded for stability and explicitness)
    NUMERIC_COLUMNS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "is_new_customer",
        "high_risk_contract",
        "avg_monthly_spend_ratio",
        "num_services"
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
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERIC_COLUMNS),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_COLUMNS)
        ],
        remainder="drop"  # Drop any columns not explicitly listed
    )

    # Build outer Pipeline: feature engineering first, then column transformations
    preprocessor = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features)),
        ("column_transformer", column_transformer)
    ])

    return preprocessor


def build_full_preprocessor_v2() -> Pipeline:
    """
    Build a full preprocessor that applies feature engineering (v2) then column transformations.

    This function creates a Pipeline that:
    1. Applies feature engineering v2 (adds only 2 non-redundant engineered features)
    2. Applies ColumnTransformer for numeric and categorical preprocessing:
       - Numeric columns: clean, impute with median, scale
       - Categorical columns: impute with most frequent, one-hot encode

    Returns
    -------
    Pipeline
        sklearn Pipeline that applies feature engineering (v2) then column transformations.
    """
    # Define column lists (hardcoded for stability and explicitness)
    NUMERIC_COLUMNS = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "avg_monthly_spend_ratio",
        "num_services"
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
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERIC_COLUMNS),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_COLUMNS)
        ],
        remainder="drop"  # Drop any columns not explicitly listed
    )

    # Build outer Pipeline: feature engineering first, then column transformations
    preprocessor = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features_v2)),
        ("column_transformer", column_transformer)
    ])

    return preprocessor
