"""Run categorical preprocessing pipeline on train/test split."""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_categorical_pipeline


def main():
    """Load, split, and preprocess categorical features."""
    print("Loading raw data...")
    df = load_raw_data()

    print("\nPerforming train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    # Define numeric columns (to exclude)
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    # Define categorical columns (all columns except numeric ones)
    categorical_columns = [col for col in X_train.columns if col not in numeric_columns]

    print(f"\nCategorical columns: {categorical_columns}")
    print(f"Number of categorical columns: {len(categorical_columns)}")

    # Select only categorical columns
    X_train_categorical = X_train[categorical_columns].copy()
    X_test_categorical = X_test[categorical_columns].copy()

    print(f"\nX_train_categorical shape before preprocessing: {X_train_categorical.shape}")
    print(f"X_test_categorical shape before preprocessing: {X_test_categorical.shape}")

    # Build the categorical pipeline
    print("\nBuilding categorical pipeline...")
    categorical_pipeline = build_categorical_pipeline()

    # Fit the pipeline on training data and transform it
    # We fit on training data only to avoid data leakage - the imputer learns
    # the most frequent values from training data, and the encoder learns the
    # categories from training data. We then apply these learned transformations
    # to test data.
    print("\nFitting pipeline on training data and transforming...")
    X_train_categorical_transformed = categorical_pipeline.fit_transform(X_train_categorical)

    # Transform test data using the fitted pipeline (do NOT fit on test data)
    # This ensures the test data is transformed using the same parameters
    # (most frequent for imputation, categories for encoding) learned from training data.
    print("Transforming test data using fitted pipeline...")
    X_test_categorical_transformed = categorical_pipeline.transform(X_test_categorical)

    print(f"\nX_train_categorical_transformed shape: {X_train_categorical_transformed.shape}")
    print(f"X_test_categorical_transformed shape: {X_test_categorical_transformed.shape}")

    # Get feature names from the pipeline
    feature_names = categorical_pipeline.get_feature_names_out()
    print(f"\nGenerated feature names ({len(feature_names)} total):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")

    # Check for NaN values
    print("\nChecking for NaN values:")
    train_has_nan = np.isnan(X_train_categorical_transformed).any()
    test_has_nan = np.isnan(X_test_categorical_transformed).any()
    print(f"  X_train has NaN: {train_has_nan}")
    print(f"  X_test has NaN: {test_has_nan}")

    # Print expansion summary
    print("\n" + "="*60)
    print("CATEGORICAL ENCODING SUMMARY")
    print("="*60)
    print(f"Original categorical columns: {len(categorical_columns)}")
    print(f"Columns after one-hot encoding: {len(feature_names)}")
    print(f"Column expansion: {len(feature_names) - len(categorical_columns)} additional columns")
    print("="*60)
    print("CATEGORICAL PREPROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
