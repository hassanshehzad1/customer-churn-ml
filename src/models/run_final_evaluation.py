"""Script for Phase 8, Part 1: Final model training and ONE-TIME test set evaluation.

This is the ONLY time we touch X_test/y_test for evaluation in the entire project.
No further tuning will be done based on test set results - this is the final assessment.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def print_confusion_matrix(cm, title="Confusion Matrix"):
    """Print a labeled confusion matrix as a 2x2 table."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"                Predicted No   Predicted Yes")
    print(f"Actual No       {cm[0,0]:^15} {cm[0,1]:^15}")
    print(f"Actual Yes      {cm[1,0]:^15} {cm[1,1]:^15}")
    print()
    print("Key:")
    print(f"  True Negative (TN):  {cm[0,0]} - Correctly predicted not churning")
    print(f"  False Positive (FP): {cm[0,1]} - Incorrectly predicted churning (false alarm)")
    print(f"  False Negative (FN): {cm[1,0]} - Incorrectly predicted not churning (missed churner)")
    print(f"  True Positive (TP):  {cm[1,1]} - Correctly predicted churning")
    print()


def compute_metrics_at_threshold(y_true, y_proba, threshold):
    """Compute metrics at a specific decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'y_pred': y_pred
    }


def main():
    print("=" * 70)
    print("Phase 8, Part 1: Final Model Training and Test Set Evaluation")
    print("=" * 70)
    print()
    print("This is the ONE AND ONLY time we evaluate on X_test/y_test.")
    print("No further tuning will be done based on these results.")
    print()

    # Step 1: Load data
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print()

    # Step 2: Split data
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print()

    # Step 3: Build the final pipeline
    print("Step 3: Building final pipeline...")
    print("  Preprocessor: build_full_preprocessor() (original, no engineered features)")
    print("  Model: LogisticRegression(max_iter=1000, random_state=42)")
    preprocessor = build_full_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    print()

    # Step 4: Fit on FULL training data (not cross-validation)
    print("Step 4: Fitting pipeline on FULL X_train/y_train...")
    print("  This is the final, single fit - no cross-validation this time.")
    pipeline.fit(X_train, y_train)
    print("  Model fitted successfully.")
    print()

    # Step 5: Get predicted probabilities on test set
    print("Step 5: Getting predicted probabilities on X_test...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
    print(f"  Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"  Mean probability: {y_proba.mean():.4f}")
    print()

    # Step 6: Compute threshold-independent metrics (ROC-AUC, PR-AUC)
    print("Step 6: Computing threshold-independent metrics...")
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print()

    # Step 7: Compute metrics at threshold 0.3 (our chosen threshold)
    print("Step 7: Computing metrics at threshold 0.3 (chosen threshold)...")
    metrics_03 = compute_metrics_at_threshold(y_test, y_proba, 0.3)
    print(f"  Accuracy:  {metrics_03['accuracy']:.4f}")
    print(f"  Precision: {metrics_03['precision']:.4f}")
    print(f"  Recall:    {metrics_03['recall']:.4f}")
    print(f"  F1:        {metrics_03['f1']:.4f}")
    print()

    # Step 8: Compute metrics at threshold 0.5 (default, for comparison)
    print("Step 8: Computing metrics at threshold 0.5 (default, for comparison)...")
    metrics_05 = compute_metrics_at_threshold(y_test, y_proba, 0.5)
    print(f"  Accuracy:  {metrics_05['accuracy']:.4f}")
    print(f"  Precision: {metrics_05['precision']:.4f}")
    print(f"  Recall:    {metrics_05['recall']:.4f}")
    print(f"  F1:        {metrics_05['f1']:.4f}")
    print()

    # Step 9: Compute and print confusion matrices
    print("Step 9: Computing confusion matrices...")
    
    cm_03 = confusion_matrix(y_test, metrics_03['y_pred'])
    print_confusion_matrix(cm_03, "Confusion Matrix at Threshold 0.3")
    
    cm_05 = confusion_matrix(y_test, metrics_05['y_pred'])
    print_confusion_matrix(cm_05, "Confusion Matrix at Threshold 0.5")
    print()

    # Step 10: Final summary and comparison
    print("=" * 70)
    print("FINAL SUMMARY AND COMPARISON")
    print("=" * 70)
    print()
    
    print("Threshold-Independent Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print()
    
    print("Metrics at Threshold 0.3 (chosen):")
    print(f"  Accuracy:  {metrics_03['accuracy']:.4f} ({metrics_03['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics_03['precision']:.4f} ({metrics_03['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics_03['recall']:.4f} ({metrics_03['recall']*100:.2f}%)")
    print(f"  F1:        {metrics_03['f1']:.4f} ({metrics_03['f1']*100:.2f}%)")
    print()
    
    print("Metrics at Threshold 0.5 (default):")
    print(f"  Accuracy:  {metrics_05['accuracy']:.4f} ({metrics_05['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics_05['precision']:.4f} ({metrics_05['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics_05['recall']:.4f} ({metrics_05['recall']*100:.2f}%)")
    print(f"  F1:        {metrics_05['f1']:.4f} ({metrics_05['f1']*100:.2f}%)")
    print()
    
    print("Difference (0.3 vs 0.5):")
    print(f"  Accuracy:  {metrics_03['accuracy'] - metrics_05['accuracy']:+.4f}")
    print(f"  Precision: {metrics_03['precision'] - metrics_05['precision']:+.4f}")
    print(f"  Recall:    {metrics_03['recall'] - metrics_05['recall']:+.4f}")
    print(f"  F1:        {metrics_03['f1'] - metrics_05['f1']:+.4f}")
    print()

    # Step 11: Plain-English explanation of confusion matrix
    print("=" * 70)
    print("PLAIN-ENGLISH EXPLANATION OF CONFUSION MATRIX (Threshold 0.3)")
    print("=" * 70)
    print()
    
    tn, fp, fn, tp = cm_03.ravel()
    total = len(y_test)
    
    print(f"Out of {total} customers in the test set:")
    print()
    print(f"  [OK] {tn} customers correctly identified as NOT churning (True Negatives)")
    print(f"  [OK] {tp} customers correctly identified as LIKELY to churn (True Positives)")
    print()
    print(f"  [X] {fp} customers incorrectly flagged as churning (False Positives)")
    print(f"    -> These are 'false alarms' - customers we'd try to retain but who")
    print(f"       wouldn't have actually left. This costs retention effort but not")
    print(f"       a lost customer.")
    print()
    print(f"  [X] {fn} customers incorrectly identified as NOT churning (False Negatives)")
    print(f"    -> These are 'missed churners' - customers who WILL leave but we didn't")
    print(f"       catch them. This is the most costly error since we lose revenue.")
    print()
    
    print(f"Summary:")
    print(f"  - We correctly caught {tp} out of {tp + fn} actual churners ({recall_score(y_test, metrics_03['y_pred'])*100:.2f}% recall)")
    print(f"  - We had {fp} false alarms out of {tp + fp} churn predictions ({precision_score(y_test, metrics_03['y_pred'])*100:.2f}% precision)")
    print(f"  - Overall accuracy: {accuracy_score(y_test, metrics_03['y_pred'])*100:.2f}%")
    print()

    print("=" * 70)
    print("Phase 8, Part 1 COMPLETE")
    print("=" * 70)
    print()
    print("CRITICAL REMINDER:")
    print("  - This is the ONLY evaluation on X_test/y_test in the entire project.")
    print("  - No further tuning will be done based on these results.")
    print("  - If results differ from expectations, we document and discuss.")
    print("  - We do NOT go back and re-tune using test set feedback (data leakage).")
    print()
    print("Next steps (Phase 8, Part 3): Save the final trained model for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
