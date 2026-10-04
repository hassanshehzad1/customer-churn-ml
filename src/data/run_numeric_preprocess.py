"""Run numeric preprocessing pipeline on train/test split."""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_numeric_pipeline


def main():
    """Load, split, and preprocess numeric features."""
    print("Loading raw data...")
    df = load_raw_data()

    print("\nPerforming train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Define numeric columns
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    print(f"\nNumeric columns: {numeric_columns}")

    # Select only numeric columns
    X_train_numeric = X_train[numeric_columns].copy()
    X_test_numeric = X_test[numeric_columns].copy()

    print(f"\nX_train_numeric shape before preprocessing: {X_train_numeric.shape}")
    print(f"X_test_numeric shape before preprocessing: {X_test_numeric.shape}")

    # Build the numeric pipeline
    print("\nBuilding numeric pipeline...")
    numeric_pipeline = build_numeric_pipeline()

    # Fit the pipeline on training data and transform it
    # We fit on training data only to avoid data leakage - the imputer learns
    # median values from training data, and the scaler learns mean/std from
    # training data. We then apply these learned transformations to test data.
    print("\nFitting pipeline on training data and transforming...")
    X_train_numeric_transformed = numeric_pipeline.fit_transform(X_train_numeric)

    # Transform test data using the fitted pipeline (do NOT fit on test data)
    # This ensures the test data is transformed using the same parameters
    # (median for imputation, mean/std for scaling) learned from training data.
    print("Transforming test data using fitted pipeline...")
    X_test_numeric_transformed = numeric_pipeline.transform(X_test_numeric)

    print(f"\nX_train_numeric_transformed shape: {X_train_numeric_transformed.shape}")
    print(f"X_test_numeric_transformed shape: {X_test_numeric_transformed.shape}")

    # Print mean and std of each column in transformed X_train
    print("\nMean and std of transformed X_train (should be ~0 and ~1):")
    X_train_transformed_df = pd.DataFrame(X_train_numeric_transformed, columns=numeric_columns)
    for col in numeric_columns:
        mean = X_train_transformed_df[col].mean()
        std = X_train_transformed_df[col].std()
        print(f"  {col}: mean={mean:.6f}, std={std:.6f}")

    # Check for NaN values
    print("\nChecking for NaN values:")
    train_has_nan = np.isnan(X_train_numeric_transformed).any()
    test_has_nan = np.isnan(X_test_numeric_transformed).any()
    print(f"  X_train has NaN: {train_has_nan}")
    print(f"  X_test has NaN: {test_has_nan}")

    print("\n" + "="*60)
    print("NUMERIC PREPROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
