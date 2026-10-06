"""Script to run cross-validation on all 4 baseline models and compare results."""

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
    print("Phase 5, Part 3: Baseline Model Comparison")
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

    # Step 3: Get all baseline models
    print("Step 3: Getting all baseline models...")
    models = get_baseline_models()
    print(f"Found {len(models)} models: {list(models.keys())}")
    print()

    # Step 4: Cross-validate each model
    print("Step 4: Running 5-fold cross-validation on each model...")
    print("Metrics: accuracy, precision, recall, f1")
    print()

    results = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for model_name, model in models.items():
        print(f"Processing: {model_name}...")

        # Build a fresh preprocessor for this model
        preprocessor = build_full_preprocessor()

        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Run cross-validation
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=metrics,
            return_train_score=False
        )

        # Store mean scores
        model_results = {}
        for metric in metrics:
            test_metric = f'test_{metric}'
            mean_score = cv_results[test_metric].mean()
            model_results[metric] = mean_score

        results[model_name] = model_results
        print(f"  Completed. Mean F1: {model_results['f1']:.4f}")
        print()

    # Step 5: Print comparison table
    print("=" * 70)
    print("BASELINE MODEL COMPARISON TABLE")
    print("=" * 70)
    print()

    # Create DataFrame for clean table display
    results_df = pd.DataFrame(results).T * 100  # Convert to percentages
    results_df = results_df.round(2)

    print(results_df.to_string())
    print()

    # Step 6: Print best model for each metric
    print("=" * 70)
    print("BEST MODEL FOR EACH METRIC")
    print("=" * 70)
    print()

    for metric in metrics:
        best_model = max(results, key=lambda x: results[x][metric])
        best_score = results[best_model][metric] * 100
        print(f"Best {metric.upper()}: {best_model} ({best_score:.2f}%)")

    print()

    # Step 7: Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    print("Why Recall matters most for this business problem:")
    print("- In churn prediction, false negatives are costly: failing to identify")
    print("  a customer who will churn means losing that customer without intervention.")
    print("- False positives (predicting churn when they won't) are less costly:")
    print("  you might offer retention incentives to a loyal customer, but that's")
    print("  just a small marketing expense, not a lost customer.")
    print("- Therefore, Recall (ability to catch all actual churners) is the priority.")
    print("- F1 balances Recall and Precision, so it's also important to ensure")
    print("  we're not just predicting everyone will churn.")
    print()

    # Find best model based on Recall and F1
    best_recall_model = max(results, key=lambda x: results[x]['recall'])
    best_f1_model = max(results, key=lambda x: results[x]['f1'])

    print(f"Best Recall: {best_recall_model} ({results[best_recall_model]['recall']*100:.2f}%)")
    print(f"Best F1: {best_f1_model} ({results[best_f1_model]['f1']*100:.2f}%)")
    print()

    if best_recall_model == best_f1_model:
        print(f"RECOMMENDATION: {best_recall_model} is the most promising model.")
        print(f"It achieves the best balance of Recall ({results[best_recall_model]['recall']*100:.2f}%)")
        print(f"and F1 ({results[best_recall_model]['f1']*100:.2f}%), making it ideal for")
        print("identifying customers at risk of churn while maintaining reasonable precision.")
    else:
        print(f"RECOMMENDATION: Compare {best_recall_model} (best Recall) vs {best_f1_model} (best F1).")
        print(f"{best_recall_model} has higher Recall ({results[best_recall_model]['recall']*100:.2f}%)")
        print(f"but {best_f1_model} has better overall balance (F1: {results[best_f1_model]['f1']*100:.2f}%).")
        print("Consider the business trade-off: maximizing churn detection vs avoiding false alarms.")

    print()
    print("=" * 70)
    print("NOTE: X_test and y_test were NOT used in this step.")
    print("Cross-validation only uses the training set (X_train, y_train).")
    print("No models were saved yet - that will happen in Phase 8.")
    print("=" * 70)
