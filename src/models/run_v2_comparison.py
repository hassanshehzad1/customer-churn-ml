"""Script to test Logistic Regression with only non-redundant engineered features."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor_v2


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 6, Part 2b: Logistic Regression with Non-Redundant Features")
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

    # Step 3: Build model and preprocessor
    print("Step 3: Building Logistic Regression with v2 preprocessor...")
    preprocessor = build_full_preprocessor_v2()
    model = LogisticRegression(max_iter=1000, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    print()

    # Step 4: Cross-validate
    print("Step 4: Running 5-fold cross-validation...")
    print("Metrics: accuracy, precision, recall, f1")
    print()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=metrics,
        return_train_score=False
    )

    # Extract mean scores
    results = {}
    for metric in metrics:
        test_metric = f'test_{metric}'
        mean_score = cv_results[test_metric].mean()
        results[metric] = mean_score

    print("Cross-validation complete!")
    print()

    # Step 5: Print current results
    print("=" * 70)
    print("PHASE 6 PART 2B RESULTS (Logistic Regression, 2 non-redundant features)")
    print("=" * 70)
    print()
    for metric in metrics:
        print(f"{metric.capitalize()}: {results[metric]*100:.2f}%")
    print()

    # Step 6: Print 3-way comparison table
    print("=" * 70)
    print("3-WAY COMPARISON TABLE")
    print("=" * 70)
    print()

    comparison_data = {
        "Phase 5 (no engineered features)": {
            "Accuracy": 80.42,
            "Precision": 65.77,
            "Recall": 54.85,
            "F1": 59.80
        },
        "Phase 6 Part 2 (all 4 features)": {
            "Accuracy": 80.58,
            "Precision": 66.91,
            "Recall": 53.18,
            "F1": 59.22
        },
        "Phase 6 Part 2b (2 non-redundant features)": {
            "Accuracy": results['accuracy'] * 100,
            "Precision": results['precision'] * 100,
            "Recall": results['recall'] * 100,
            "F1": results['f1'] * 100
        }
    }

    comparison_df = pd.DataFrame(comparison_data).T
    print(comparison_df.round(2).to_string())
    print()

    # Step 7: Find best version
    print("=" * 70)
    print("BEST VERSION FOR EACH METRIC")
    print("=" * 70)
    print()

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        best_version = max(comparison_data, key=lambda x: comparison_data[x][metric])
        best_score = comparison_data[best_version][metric]
        print(f"Best {metric}: {best_version} ({best_score:.2f}%)")

    print()

    # Step 8: Compare with Phase 5 baseline
    print("=" * 70)
    print("IMPROVEMENT OVER PHASE 5 BASELINE")
    print("=" * 70)
    print()

    recall_improvement = (results['recall'] - 0.5485) * 100
    f1_improvement = (results['f1'] - 0.5980) * 100

    print(f"Recall Improvement: {recall_improvement:+.2f} percentage points")
    print(f"F1 Improvement: {f1_improvement:+.2f} percentage points")
    print()

    if recall_improvement > 0 and f1_improvement > 0:
        print("CONCLUSION: Phase 6 Part 2b IMPROVED both Recall and F1 over Phase 5.")
    elif recall_improvement > 0:
        print("CONCLUSION: Phase 6 Part 2b IMPROVED Recall but not F1 over Phase 5.")
    elif f1_improvement > 0:
        print("CONCLUSION: Phase 6 Part 2b IMPROVED F1 but not Recall over Phase 5.")
    else:
        print("CONCLUSION: Phase 6 Part 2b did NOT improve Recall or F1 over Phase 5.")

    print()

    # Step 9: Compare with Phase 6 Part 2
    print("=" * 70)
    print("COMPARISON WITH PHASE 6 PART 2 (all 4 features)")
    print("=" * 70)
    print()

    recall_diff_v2 = (results['recall'] - 0.5318) * 100
    f1_diff_v2 = (results['f1'] - 0.5922) * 100

    print(f"Recall Difference: {recall_diff_v2:+.2f} percentage points")
    print(f"F1 Difference: {f1_diff_v2:+.2f} percentage points")
    print()

    if recall_diff_v2 > 0 and f1_diff_v2 > 0:
        print("CONCLUSION: Removing redundant features IMPROVED both Recall and F1.")
    elif recall_diff_v2 > 0:
        print("CONCLUSION: Removing redundant features IMPROVED Recall but not F1.")
    elif f1_diff_v2 > 0:
        print("CONCLUSION: Removing redundant features IMPROVED F1 but not Recall.")
    else:
        print("CONCLUSION: Removing redundant features did NOT improve Recall or F1.")

    print()
    print("=" * 70)
    print("NOTE: X_test and y_test were NOT used in this step.")
    print("Cross-validation only uses the training set (X_train, y_train).")
    print("=" * 70)
