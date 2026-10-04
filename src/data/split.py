"""Functions for splitting data into train and test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COLUMN, ID_COLUMN


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into train and test sets with stratification.

    This function creates a train/test split with stratification on the target variable
    to ensure balanced class distribution in both sets. No preprocessing, scaling,
    encoding, or imputing is performed - just a raw split.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to split.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target values (1 for "Yes", 0 for "No").
    y_test : pd.Series
        Test target values (1 for "Yes", 0 for "No").
    """
    # Create y: map "Yes" to 1, "No" to 0
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Create X: drop TARGET_COLUMN and ID_COLUMN, keep everything else as-is
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    # Perform stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Print churn rates for stratification verification
    train_churn_rate = y_train.mean() * 100
    test_churn_rate = y_test.mean() * 100
    print(f"\nChurn rate in training set: {train_churn_rate:.2f}%")
    print(f"Churn rate in test set: {test_churn_rate:.2f}%")

    return X_train, X_test, y_train, y_test
