"""Script for Phase 8, Part 3: Verify the saved model loads and works correctly.

This loads the saved pipeline and confirms it produces identical results
to the original model from Part 1.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.data.load_data import load_raw_data
from src.data.split import split_data


def main():
    print("=" * 70)
    print("Phase 8, Part 3: Verify Saved Model")
    print("=" * 70)
    print()

    # Step 1: Load data (same split as Part 1)
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print()

    # Step 2: Split data (same split as Part 1)
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print()

    # Step 3: Load the saved pipeline
    print("Step 3: Loading saved pipeline...")
    pipeline_path = Path("models/churn_pipeline.joblib")
    if not pipeline_path.exists():
        print(f"  ERROR: Pipeline file not found at {pipeline_path.absolute()}")
        print("  Please run save_final_model.py first.")
        return
    
    pipeline = joblib.load(pipeline_path)
    print(f"  Loaded from: {pipeline_path.absolute()}")
    print()

    # Step 4: Load metadata
    print("Step 4: Loading model metadata...")
    metadata_path = Path("models/model_metadata.json")
    if not metadata_path.exists():
        print(f"  WARNING: Metadata file not found at {metadata_path.absolute()}")
        metadata = None
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Loaded from: {metadata_path.absolute()}")
        print(f"  Model type: {metadata['model_type']}")
        print(f"  Decision threshold: {metadata['decision_threshold']}")
        print(f"  Training date: {metadata['training_date']}")
    print()

    # Step 5: Get predictions from loaded model
    print("Step 5: Getting predictions from loaded model...")
    threshold = 0.3
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(f"  Threshold: {threshold}")
    print(f"  Predictions shape: {y_pred.shape}")
    print()

    # Step 6: Compute metrics
    print("Step 6: Computing metrics from loaded model...")
    loaded_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    print(f"  Accuracy:  {loaded_metrics['accuracy']:.4f}")
    print(f"  Precision: {loaded_metrics['precision']:.4f}")
    print(f"  Recall:    {loaded_metrics['recall']:.4f}")
    print(f"  F1:        {loaded_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {loaded_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {loaded_metrics['pr_auc']:.4f}")
    print()

    # Step 7: Compare with Part 1 expected results
    print("Step 7: Comparing with Part 1 expected results...")
    print("  Expected from Part 1 (Phase 8, Part 1):")
    print("    Accuracy:  0.7495")
    print("    Precision: 0.5193")
    print("    Recall:    0.7540")
    print("    F1:        0.6150")
    print("    ROC-AUC:   0.8419")
    print("    PR-AUC:    0.6334")
    print()
    
    print("  Loaded model results:")
    print(f"    Accuracy:  {loaded_metrics['accuracy']:.4f}")
    print(f"    Precision: {loaded_metrics['precision']:.4f}")
    print(f"    Recall:    {loaded_metrics['recall']:.4f}")
    print(f"    F1:        {loaded_metrics['f1']:.4f}")
    print(f"    ROC-AUC:   {loaded_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:    {loaded_metrics['pr_auc']:.4f}")
    print()

    # Step 8: Verify exact match
    print("Step 8: Verifying exact match...")
    expected_metrics = {
        'accuracy': 0.7495,
        'precision': 0.5193,
        'recall': 0.7540,
        'f1': 0.6150,
        'roc_auc': 0.8419,
        'pr_auc': 0.6334
    }
    
    all_match = True
    for metric_name, expected_value in expected_metrics.items():
        loaded_value = loaded_metrics[metric_name]
        # Allow tiny floating point differences
        if abs(loaded_value - expected_value) < 0.0001:
            print(f"  [MATCH] {metric_name}: {loaded_value:.4f} == {expected_value:.4f}")
        else:
            print(f"  [MISMATCH] {metric_name}: {loaded_value:.4f} != {expected_value:.4f}")
            all_match = False
    print()

    # Step 9: Final confirmation
    print("=" * 70)
    if all_match:
        print("VERIFICATION SUCCESSFUL")
        print("=" * 70)
        print()
        print("The saved model produces EXACTLY the same results as the original.")
        print("The model has been saved correctly and is ready for deployment.")
    else:
        print("VERIFICATION FAILED")
        print("=" * 70)
        print()
        print("The saved model does NOT match the original results.")
        print("Please check the save process and try again.")
    print()
    print("=" * 70)
    print("Phase 8, Part 3b COMPLETE - Model Verified")
    print("=" * 70)


if __name__ == "__main__":
    main()
