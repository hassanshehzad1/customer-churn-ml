"""Run full preprocessing pipeline (numeric + categorical) on train/test split."""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def main():
    """Load, split, and preprocess all features (numeric + categorical)."""
    print("Loading raw data...")
    df = load_raw_data()

    print("\nPerforming train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"\nX_train shape before preprocessing: {X_train.shape}")
    print(f"X_test shape before preprocessing: {X_test.shape}")

    # Build the full preprocessor
    print("\nBuilding full preprocessor (numeric + categorical)...")
    full_preprocessor = build_full_preprocessor()

    # Fit the preprocessor on training data ONLY and transform it
    # CRITICAL: We fit on training data only to avoid data leakage.
    # If we fit on test data or the full dataset, the imputer would learn
    # median/most-frequent values from test data, and the scaler would learn
    # mean/std from test data. This would give the model information about
    # the test set during training, leading to overly optimistic performance
    # estimates that won't generalize to new data.
    print("\nFitting preprocessor on training data ONLY and transforming...")
    X_train_transformed = full_preprocessor.fit_transform(X_train)

    # Transform test data using the fitted preprocessor (do NOT fit on test data)
    # This ensures the test data is transformed using the same parameters
    # learned from training data only.
    print("Transforming test data using fitted preprocessor...")
    X_test_transformed = full_preprocessor.transform(X_test)

    print(f"\nX_train shape after preprocessing: {X_train_transformed.shape}")
    print(f"X_test shape after preprocessing: {X_test_transformed.shape}")

    # Get feature count from the transformed data shape
    actual_features = X_train_transformed.shape[1]
    print(f"\nTotal output features: {actual_features}")

    # Check for NaN values
    print("\nChecking for NaN values:")
    train_has_nan = np.isnan(X_train_transformed).any()
    test_has_nan = np.isnan(X_test_transformed).any()
    print(f"  X_train has NaN: {train_has_nan}")
    print(f"  X_test has NaN: {test_has_nan}")

    # Verify feature count matches expectation
    expected_features = 45  # 4 numeric + 41 categorical
    print("\n" + "="*60)
    print("FULL PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Expected features (4 numeric + 41 categorical): {expected_features}")
    print(f"Actual features: {actual_features}")
    print(f"Match: {expected_features == actual_features}")
    print("="*60)
    print("FULL PREPROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
