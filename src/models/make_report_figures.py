"""Generate report figures for the final model.

This script loads the saved pipeline, computes metrics on the test set,
and generates confusion matrix, ROC curve, and precision-recall curve figures.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

from src.data.load_data import load_raw_data
from src.data.split import split_data


def main():
    print("=" * 70)
    print("Generating Report Figures for Final Model")
    print("=" * 70)

    # Load model metadata
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    expected_threshold = metadata["decision_threshold"]
    expected_metrics = metadata["test_set_metrics"]

    print(f"\nExpected threshold: {expected_threshold}")
    print("Expected metrics from metadata:")
    for metric, value in expected_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Load the saved pipeline
    print("\nLoading saved pipeline...")
    pipeline = joblib.load("models/churn_pipeline.joblib")
    print("Pipeline loaded successfully.")

    # Load and split data
    print("\nLoading and splitting data...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    # Get predictions
    print("\nComputing predictions...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= expected_threshold).astype(int)

    # Compute metrics
    print("\nComputing metrics...")
    computed_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba)
    }

    # Compare metrics
    print("\n" + "=" * 70)
    print("METRIC COMPARISON (Computed vs Metadata)")
    print("=" * 70)
    print(f"{'Metric':<15} {'Computed':<15} {'Expected':<15} {'Match':<10}")
    print("-" * 70)

    all_match = True
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        computed = computed_metrics[metric]
        expected = expected_metrics[metric]
        match = abs(computed - expected) < 1e-6
        match_str = "PASS" if match else "FAIL"
        print(f"{metric:<15} {computed:.4f}          {expected:.4f}          {match_str:<10}")
        if not match:
            all_match = False

    print("=" * 70)

    if not all_match:
        print("\nERROR: Metrics do not match! Stopping without saving figures.")
        print("This indicates the saved model or data split is different from Phase 8.")
        return

    print("\nAll metrics match! Proceeding to generate figures...")

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize by row to get percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotated heatmap
    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                xticklabels=['No churn', 'Churn'],
                yticklabels=['No churn', 'Churn'],
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Threshold = {expected_threshold}, Test Set)\n',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/confusion_matrix.png")

    # Generate ROC curve
    print("\nGenerating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = computed_metrics["roc_auc"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    # Find the operating point at threshold 0.3
    threshold_idx = np.argmin(np.abs(thresholds - expected_threshold))
    ax.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=10,
            label=f'Threshold = {expected_threshold}')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/roc_curve.png")

    # Generate Precision-Recall curve
    print("\nGenerating Precision-Recall curve...")
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    pr_auc = computed_metrics["pr_auc"]
    baseline_rate = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2,
            label=f'PR Curve (AP = {pr_auc:.4f})')
    ax.axhline(y=baseline_rate, color='r', linestyle='--', linewidth=1,
               label=f'Baseline (Churn Rate = {baseline_rate:.3f})')

    # Find the operating point at threshold 0.3
    # Need to find the corresponding point in the precision-recall curve
    # We use the y_proba to find which threshold corresponds to our operating point
    threshold_idx_pr = np.argmin(np.abs(thresholds_pr - expected_threshold))
    ax.plot(recall[threshold_idx_pr], precision[threshold_idx_pr], 'ro',
            markersize=10, label=f'Threshold = {expected_threshold}')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/pr_curve.png")

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
