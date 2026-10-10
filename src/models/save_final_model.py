"""Script for Phase 8, Part 3: Save the final trained pipeline to disk.

This saves the fitted pipeline (preprocessor + LogisticRegression) trained on
the full training set, along with metadata about the model and its performance.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
from src.features.preprocessing import build_full_preprocessor


def main():
    print("=" * 70)
    print("Phase 8, Part 3: Save Final Trained Pipeline")
    print("=" * 70)
    print()

    # Step 1: Load data
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print()

    # Step 2: Split data
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print()

    # Step 3: Build and fit the final pipeline
    print("Step 3: Building and fitting final pipeline...")
    print("  Preprocessor: build_full_preprocessor()")
    print("  Model: LogisticRegression(max_iter=1000, random_state=42)")
    preprocessor = build_full_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    print("  Model fitted successfully.")
    print()

    # Step 4: Compute test set metrics (at threshold 0.3)
    print("Step 4: Computing test set metrics at threshold 0.3...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print()

    # Step 5: Create models directory if it doesn't exist
    print("Step 5: Creating models directory...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"  Directory: {models_dir.absolute()}")
    print()

    # Step 6: Save the pipeline
    print("Step 6: Saving fitted pipeline...")
    pipeline_path = models_dir / "churn_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    pipeline_size = pipeline_path.stat().st_size
    print(f"  Saved to: {pipeline_path.absolute()}")
    print(f"  File size: {pipeline_size:,} bytes ({pipeline_size / 1024:.2f} KB)")
    print()

    # Step 7: Create and save metadata
    print("Step 7: Creating and saving model metadata...")
    metadata = {
        'model_type': 'LogisticRegression',
        'model_params': {
            'max_iter': 1000,
            'random_state': 42
        },
        'preprocessor': 'build_full_preprocessor() (original, no engineered features)',
        'decision_threshold': 0.3,
        'training_date': datetime.now().isoformat(),
        'train_set_size': len(X_train),
        'test_set_size': len(X_test),
        'test_set_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc']
        },
        'notes': 'Final model trained on full training set. Threshold 0.3 chosen based on Phase 7 threshold tuning. Test set metrics from Phase 8, Part 1 evaluation.'
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    metadata_size = metadata_path.stat().st_size
    print(f"  Saved to: {metadata_path.absolute()}")
    print(f"  File size: {metadata_size:,} bytes ({metadata_size / 1024:.2f} KB)")
    print()

    # Step 8: Print confirmation
    print("=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Pipeline file: {pipeline_path.absolute()}")
    print(f"  Size: {pipeline_size:,} bytes ({pipeline_size / 1024:.2f} KB)")
    print()
    print(f"Metadata file: {metadata_path.absolute()}")
    print(f"  Size: {metadata_size:,} bytes ({metadata_size / 1024:.2f} KB)")
    print()
    print("Metadata contents:")
    print(json.dumps(metadata, indent=2))
    print()
    print("=" * 70)
    print("Phase 8, Part 3a COMPLETE - Model Saved")
    print("=" * 70)
    print()
    print("Next step: Run verify_saved_model.py to confirm the saved model")
    print("produces identical results to the original.")
    print("=" * 70)


if __name__ == "__main__":
    main()
