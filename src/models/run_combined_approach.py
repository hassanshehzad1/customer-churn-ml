"""Combined approach: class_weight='balanced' + threshold tuning.

This script combines class weighting with threshold tuning to see if both
techniques together achieve better results than either alone.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def main():
    """Run combined approach analysis on validation set."""
    print("=" * 80)
    print("Phase 7, Part 2b: Combined Approach (Class Weight + Threshold Tuning)")
    print("=" * 80)
    print()

    # Step 1: Load data and split into train/test (test set MUST remain untouched)
    print("Step 1: Loading data and splitting into train/test...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print()

    # Step 2: Further split X_train/y_train into inner train and validation sets
    print("Step 2: Splitting training data into inner train/validation (80/20)...")
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"X_train_inner shape: {X_train_inner.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train_inner shape: {y_train_inner.shape}")
    print(f"y_val shape: {y_val.shape}")
    print()

    # Step 3: Build pipeline with class_weight='balanced'
    print("Step 3: Building pipeline with build_full_preprocessor() + LogisticRegression(class_weight='balanced')...")
    preprocessor = build_full_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    print("Pipeline built successfully.")
    print()

    # Step 4: Fit pipeline on inner training set ONLY
    print("Step 4: Fitting pipeline on X_train_inner/y_train_inner...")
    pipeline.fit(X_train_inner, y_train_inner)
    print("Pipeline fitted successfully.")
    print()

    # Step 5: Get predicted probabilities on validation set
    print("Step 5: Getting predicted probabilities on X_val...")
    y_proba = pipeline.predict_proba(X_val)[:, 1]  # Probability of class 1 (churn)
    print(f"Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print()

    # Step 6: Evaluate different thresholds for class-weighted model
    print("Step 6: Evaluating thresholds [0.3, 0.4, 0.5, 0.6, 0.7] for class-weighted model...")
    print()

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for threshold in thresholds:
        # Convert probabilities to binary predictions using threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    # Step 7: Print threshold table for class-weighted model
    print("=" * 80)
    print("Class-Weighted Model: Threshold Analysis")
    print("=" * 80)
    print()
    print(f"{'Threshold':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['threshold']:<10.1f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    print()

    # Step 8: Identify best threshold for class-weighted model
    best_f1 = max(results, key=lambda x: x["f1"])
    print(f"Best F1 for class-weighted model: {best_f1['f1']:.4f} at threshold {best_f1['threshold']:.1f}")
    print()

    # Step 9: 4-way comparison
    print("=" * 80)
    print("4-WAY COMPARISON")
    print("=" * 80)
    print()

    # Phase 5 results (no weight, threshold 0.5) - from Phase 5 CV
    phase5_results = {
        "accuracy": 0.8042,
        "precision": 0.6577,
        "recall": 0.5485,
        "f1": 0.5980
    }

    # Phase 7 Part 1 results (no weight, threshold 0.3) - from threshold analysis
    phase7_part1_results = {
        "accuracy": 0.7764,
        "precision": 0.5577,
        "recall": 0.7592,
        "f1": 0.6431
    }

    # Phase 7 Part 2 results (class_weight='balanced', threshold 0.5) - from class weight comparison
    phase7_part2_results = {
        "accuracy": 0.7529,
        "precision": 0.5228,
        "recall": 0.8047,
        "f1": 0.6337
    }

    # Phase 7 Part 2b results (class_weight='balanced' + best threshold)
    phase7_part2b_results = {
        "accuracy": best_f1['accuracy'],
        "precision": best_f1['precision'],
        "recall": best_f1['recall'],
        "f1": best_f1['f1']
    }

    # Build comparison DataFrame
    comparison_data = {
        "Phase 5 (no weight, threshold 0.5)": phase5_results,
        "Phase 7 Part 1 (no weight, threshold 0.3)": phase7_part1_results,
        "Phase 7 Part 2 (class_weight, threshold 0.5)": phase7_part2_results,
        f"Phase 7 Part 2b (class_weight, threshold {best_f1['threshold']:.1f})": phase7_part2b_results
    }

    comparison_df = pd.DataFrame(comparison_data).T * 100  # Convert to percentages
    comparison_df = comparison_df.round(2)

    print(comparison_df.to_string())
    print()

    # Step 10: Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    print(f"Best combined approach: class_weight='balanced' + threshold={best_f1['threshold']:.1f}")
    print(f"  Precision: {best_f1['precision']:.4f} ({best_f1['precision']*100:.2f}%)")
    print(f"  Recall:    {best_f1['recall']:.4f} ({best_f1['recall']*100:.2f}%)")
    print(f"  F1 Score:  {best_f1['f1']:.4f} ({best_f1['f1']*100:.2f}%)")
    print()

    # Compare to Phase 7 Part 1 (threshold tuning alone)
    print("Comparison to threshold tuning alone (Phase 7 Part 1):")
    recall_diff = best_f1['recall'] - phase7_part1_results['recall']
    precision_diff = best_f1['precision'] - phase7_part1_results['precision']
    f1_diff = best_f1['f1'] - phase7_part1_results['f1']
    print(f"  Recall difference:    {recall_diff:+.4f} ({recall_diff*100:+.2f} percentage points)")
    print(f"  Precision difference: {precision_diff:+.4f} ({precision_diff*100:+.2f} percentage points)")
    print(f"  F1 difference:        {f1_diff:+.4f} ({f1_diff*100:+.2f} percentage points)")
    print()

    # Compare to Phase 7 Part 2 (class weighting alone)
    print("Comparison to class weighting alone (Phase 7 Part 2):")
    recall_diff_2 = best_f1['recall'] - phase7_part2_results['recall']
    precision_diff_2 = best_f1['precision'] - phase7_part2_results['precision']
    f1_diff_2 = best_f1['f1'] - phase7_part2_results['f1']
    print(f"  Recall difference:    {recall_diff_2:+.4f} ({recall_diff_2*100:+.2f} percentage points)")
    print(f"  Precision difference: {precision_diff_2:+.4f} ({precision_diff_2*100:+.2f} percentage points)")
    print(f"  F1 difference:        {f1_diff_2:+.4f} ({f1_diff_2*100:+.2f} percentage points)")
    print()

    # Step 11: Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if best_f1['f1'] > phase7_part1_results['f1'] and best_f1['f1'] > phase7_part2_results['f1']:
        print("Combining both techniques (class weighting + threshold tuning) achieves the best F1 score.")
        print("This indicates that the two approaches are complementary and work well together.")
    elif best_f1['f1'] > phase7_part1_results['f1']:
        print("Combining both techniques beats threshold tuning alone but is close to class weighting alone.")
    elif best_f1['f1'] > phase7_part2_results['f1']:
        print("Combining both techniques beats class weighting alone but is close to threshold tuning alone.")
    else:
        print("Neither approach alone is clearly beaten by the combination.")

    print()
    print("=" * 80)
    print("NOTE: X_test and y_test were NOT touched in this analysis.")
    print("=" * 80)


if __name__ == "__main__":
    main()
