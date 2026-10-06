"""Script to run cross-validation on a single model (Logistic Regression)."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor
from src.models.train import get_baseline_models


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5, Part 2: Single Model Cross-Validation")
    print("=" * 70)
    print()

    # Step 1: Load data
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print()

    # Step 2: Split data (train/test split)
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print()

    # Step 3: Get Logistic Regression model
    print("Step 3: Getting Logistic Regression model...")
    models = get_baseline_models()
    logistic_regression = models["Logistic Regression"]
    print(f"Model type: {type(logistic_regression)}")
    print()

    # Step 4: Build preprocessor
    print("Step 4: Building full preprocessor...")
    preprocessor = build_full_preprocessor()
    print(f"Preprocessor type: {type(preprocessor)}")
    print()

    # Step 5: Create pipeline
    print("Step 5: Creating sklearn Pipeline...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", logistic_regression)
    ])
    print(f"Pipeline: {pipeline}")
    print()

    # Step 6: Run 5-fold cross-validation on X_train/y_train ONLY
    print("Step 6: Running 5-fold cross-validation on training set...")
    print("Metrics: accuracy, precision, recall, f1")
    print()

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=False
    )

    # Step 7: Print each fold's scores
    print("=" * 70)
    print("CROSS-VALIDATION RESULTS: Fold-by-Fold Scores")
    print("=" * 70)
    print()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        test_metric = f'test_{metric}'
        print(f"{metric.upper()}:")
        for fold_idx, score in enumerate(cv_results[test_metric], 1):
            print(f"  Fold {fold_idx}: {score:.4f}")
        print()

    # Step 8: Print mean and std across folds
    print("=" * 70)
    print("CROSS-VALIDATION RESULTS: Mean and Std Across Folds")
    print("=" * 70)
    print()

    for metric in metrics:
        test_metric = f'test_{metric}'
        mean_score = cv_results[test_metric].mean()
        std_score = cv_results[test_metric].std()
        print(f"{metric.upper()}:")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std:  {std_score:.4f}")
        print()

    print("=" * 70)
    print("NOTE: X_test and y_test were NOT used in this step.")
    print("Cross-validation only uses the training set (X_train, y_train).")
    print("=" * 70)
