"""Class weight comparison for Logistic Regression model.

This script compares three approaches:
1. Phase 5: No class weight, threshold 0.5 (baseline)
2. Phase 7 Part 1: No class weight, threshold 0.3 (threshold-tuned)
3. Phase 7 Part 2: class_weight='balanced', threshold 0.5 (class-weighted)

The goal is to see if class weighting achieves similar recall gains to threshold tuning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def main():
    """Run class weight comparison on training set."""
    print("=" * 80)
    print("Phase 7, Part 2: Class Weight Comparison")
    print("=" * 80)
    print()

    # Step 1: Load data and split into train/test (test set MUST remain untouched)
    print("Step 1: Loading data and splitting into train/test...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print()

    # Step 2: Build pipeline with class_weight='balanced'
    print("Step 2: Building pipeline with build_full_preprocessor() + LogisticRegression(class_weight='balanced')...")
    preprocessor = build_full_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    print("Pipeline built successfully.")
    print()

    # Step 3: Run 5-fold cross-validation on training set
    print("Step 3: Running 5-fold cross-validation on X_train/y_train...")
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

    # Step 4: Compute mean scores for class_weight='balanced'
    class_weight_results = {}
    for metric in metrics:
        test_metric = f'test_{metric}'
        mean_score = cv_results[test_metric].mean()
        class_weight_results[metric] = mean_score

    print("Class Weight='balanced' Results (5-fold CV mean):")
    for metric in metrics:
        print(f"  {metric.capitalize()}: {class_weight_results[metric]:.4f}")
    print()

    # Step 5: Create comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    # Phase 5 results (from Phase 5.txt, no class weight, threshold 0.5)
    phase5_results = {
        "accuracy": 0.8042,
        "precision": 0.6577,
        "recall": 0.5485,
        "f1": 0.5980
    }

    # Phase 7 Part 1 results (from threshold analysis, no class weight, threshold 0.3)
    phase7_part1_results = {
        "accuracy": 0.7764,
        "precision": 0.5577,
        "recall": 0.7592,
        "f1": 0.6431
    }

    # Phase 7 Part 2 results (current, class_weight='balanced', threshold 0.5)
    phase7_part2_results = class_weight_results

    # Build comparison DataFrame
    comparison_data = {
        "Phase 5 (no weight, threshold 0.5)": phase5_results,
        "Phase 7 Part 1 (no weight, threshold 0.3)": phase7_part1_results,
        "Phase 7 Part 2 (class_weight='balanced', threshold 0.5)": phase7_part2_results
    }

    comparison_df = pd.DataFrame(comparison_data).T * 100  # Convert to percentages
    comparison_df = comparison_df.round(2)

    print(comparison_df.to_string())
    print()

    # Step 6: Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Compare recall gains
    phase5_recall = phase5_results['recall']
    phase7_part1_recall = phase7_part1_results['recall']
    phase7_part2_recall = phase7_part2_results['recall']

    recall_gain_threshold = phase7_part1_recall - phase5_recall
    recall_gain_class_weight = phase7_part2_recall - phase5_recall

    print(f"Recall gain from threshold tuning (0.5 -> 0.3): {recall_gain_threshold:.4f} (+{recall_gain_threshold*100:.2f} percentage points)")
    print(f"Recall gain from class weighting: {recall_gain_class_weight:.4f} (+{recall_gain_class_weight*100:.2f} percentage points)")
    print()

    # Compare precision impact
    phase5_precision = phase5_results['precision']
    phase7_part1_precision = phase7_part1_results['precision']
    phase7_part2_precision = phase7_part2_results['precision']

    precision_loss_threshold = phase5_precision - phase7_part1_precision
    precision_loss_class_weight = phase5_precision - phase7_part2_precision

    print(f"Precision loss from threshold tuning: {precision_loss_threshold:.4f} ({precision_loss_threshold*100:.2f} percentage points)")
    print(f"Precision loss from class weighting: {precision_loss_class_weight:.4f} ({precision_loss_class_weight*100:.2f} percentage points)")
    print()

    # Compare F1
    phase5_f1 = phase5_results['f1']
    phase7_part1_f1 = phase7_part1_results['f1']
    phase7_part2_f1 = phase7_part2_results['f1']

    print(f"F1 Score - Phase 5: {phase5_f1:.4f}")
    print(f"F1 Score - Phase 7 Part 1 (threshold tuning): {phase7_part1_f1:.4f}")
    print(f"F1 Score - Phase 7 Part 2 (class weighting): {phase7_part2_f1:.4f}")
    print()

    # Step 7: Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if recall_gain_class_weight >= recall_gain_threshold:
        print("Class weighting achieves similar or better recall gains than threshold tuning.")
    else:
        print("Threshold tuning achieves better recall gains than class weighting.")

    if phase7_part1_f1 >= phase7_part2_f1:
        print("Threshold tuning provides a better precision/recall balance (higher F1).")
        print("For our business case where false negatives are costly, threshold tuning at 0.3")
        print("is preferred because it catches more churners while maintaining better overall balance.")
    else:
        print("Class weighting provides a better precision/recall balance (higher F1).")
        print("This approach may be preferred if you want a more balanced model without")
        print("manually tuning the threshold.")

    print()
    print("=" * 80)
    print("NOTE: X_test and y_test were NOT used in this analysis.")
    print("Cross-validation only uses the training set (X_train, y_train).")
    print("=" * 80)


if __name__ == "__main__":
    main()
