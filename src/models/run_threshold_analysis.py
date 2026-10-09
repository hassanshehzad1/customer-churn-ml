"""Threshold analysis for Logistic Regression model.

This script explores different decision thresholds to find the optimal balance
between precision and recall. It uses a validation set carved out of the training
data, keeping the test set completely untouched.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def main():
    """Run threshold analysis on validation set."""
    print("=" * 80)
    print("Phase 7, Part 1: Threshold Analysis")
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

    # Step 3: Build pipeline with original preprocessor + Logistic Regression
    print("Step 3: Building pipeline with build_full_preprocessor() + LogisticRegression...")
    preprocessor = build_full_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
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

    # Step 6: Evaluate different thresholds
    print("Step 6: Evaluating thresholds [0.1, 0.2, ..., 0.9]...")
    print()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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

    # Step 7: Print results table
    print("=" * 80)
    print("Threshold Analysis Results")
    print("=" * 80)
    print()
    print(f"{'Threshold':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['threshold']:<10.1f} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    print()

    # Step 7.5: Plot precision-recall trade-off
    print("Step 7.5: Plotting precision-recall trade-off...")
    print()

    # Extract values for plotting
    thresh_vals = [r['threshold'] for r in results]
    prec_vals = [r['precision'] for r in results]
    rec_vals = [r['recall'] for r in results]
    f1_vals = [r['f1'] for r in results]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresh_vals, prec_vals, 'b-', linewidth=2, label='Precision')
    plt.plot(thresh_vals, rec_vals, 'r-', linewidth=2, label='Recall')
    plt.plot(thresh_vals, f1_vals, 'g-', linewidth=2, label='F1 Score')

    # Add vertical reference lines
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default threshold (0.5)')
    plt.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Business-friendly (0.3)')

    # Labels and title
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision-Recall Trade-off Across Decision Thresholds', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 1.0)

    # Save plot
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "threshold_tradeoff.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {plot_path}")
    print()

    # Step 8: Identify best thresholds
    print("=" * 80)
    print("Best Thresholds")
    print("=" * 80)
    print()

    # Best F1
    best_f1 = max(results, key=lambda x: x["f1"])
    print(f"Best F1 Score: {best_f1['f1']:.4f} at threshold {best_f1['threshold']:.1f}")
    print()

    # Best Recall with Precision >= 40%
    candidates = [r for r in results if r["precision"] >= 0.40]
    if candidates:
        best_recall = max(candidates, key=lambda x: x["recall"])
        print(f"Best Recall with Precision >= 40%: {best_recall['recall']:.4f} at threshold {best_recall['threshold']:.1f}")
        print(f"  (Precision at this threshold: {best_recall['precision']:.4f})")
    else:
        print("No threshold achieved Precision >= 40%")
    print()

    print("=" * 80)
    print("Note: X_test and y_test were NOT touched in this analysis.")
    print("=" * 80)
    print()

    # Step 9: Business recommendation
    print("=" * 80)
    print("Business Recommendation")
    print("=" * 80)
    print()
    print("For churn prediction, false negatives (missing actual churners) are more costly")
    print("than false positives (unnecessary retention outreach). A lower threshold")
    print("maximizes recall to catch more at-risk customers, even if it means some")
    print("false alarms.")
    print()
    print("Based on the trade-off analysis, threshold=0.3 offers a good balance:")
    print()

    # Find metrics at threshold 0.3
    thresh_0_3 = next(r for r in results if r['threshold'] == 0.3)
    print(f"Threshold 0.3 metrics:")
    print(f"  Precision: {thresh_0_3['precision']:.4f} ({thresh_0_3['precision']*100:.2f}%)")
    print(f"  Recall:    {thresh_0_3['recall']:.4f} ({thresh_0_3['recall']*100:.2f}%)")
    print(f"  F1 Score:  {thresh_0_3['f1']:.4f} ({thresh_0_3['f1']*100:.2f}%)")
    print()
    print("Compared to default threshold 0.5:")
    thresh_0_5 = next(r for r in results if r['threshold'] == 0.5)
    print(f"  Recall improvement: {thresh_0_3['recall'] - thresh_0_5['recall']:.4f} (+{(thresh_0_3['recall'] - thresh_0_5['recall'])*100:.2f} percentage points)")
    print(f"  Precision decrease:  {thresh_0_5['precision'] - thresh_0_3['precision']:.4f} ({(thresh_0_5['precision'] - thresh_0_3['precision'])*100:.2f} percentage points)")
    print()
    print("RECOMMENDATION: Use threshold=0.3 for production deployment.")
    print("=" * 80)


if __name__ == "__main__":
    main()
