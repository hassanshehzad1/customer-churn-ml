"""MLflow experiment tracking for churn prediction model comparisons.

This script retroactively documents Phase 5-7 model comparisons in MLflow,
providing a proper tracking system for the experiments that were run.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def main():
    """Run MLflow experiment tracking for all model configurations."""
    print("=" * 70)
    print("Phase 11, Part 1: MLflow Experiment Tracking")
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

    # Step 3: Set up MLflow experiment
    print("Step 3: Setting up MLflow experiment...")
    experiment_name = "churn_prediction_baseline_comparison_v2"
    mlflow.set_experiment(experiment_name)
    print(f"Experiment: {experiment_name}")
    print()

    # Step 4: Run MLflow experiments for 3 configurations
    print("Step 4: Running MLflow experiments for 3 configurations...")
    print()

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Configuration A: Logistic Regression, default threshold 0.5, no class_weight (Phase 5 baseline)
    print("Configuration A: Logistic Regression, no class_weight, threshold 0.5 (Phase 5 baseline)")
    with mlflow.start_run(run_name="config_a_baseline_no_class_weight"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "None")
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("evaluation_method", "5-fold CV at threshold 0.5")
        mlflow.log_param("preprocessor", "original")

        preprocessor = build_full_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=metrics,
            return_train_score=False
        )

        for metric in metrics:
            test_metric = f'test_{metric}'
            mean_score = cv_results[test_metric].mean()
            mlflow.log_metric(metric, mean_score)

        print(f"  Logged: accuracy={cv_results['test_accuracy'].mean():.4f}, "
              f"precision={cv_results['test_precision'].mean():.4f}, "
              f"recall={cv_results['test_recall'].mean():.4f}, "
              f"f1={cv_results['test_f1'].mean():.4f}")
    print()

    # Configuration B: Logistic Regression, class_weight='balanced', threshold 0.5 (Phase 7 Part 2)
    print("Configuration B: Logistic Regression, class_weight='balanced', threshold 0.5 (Phase 7 Part 2)")
    with mlflow.start_run(run_name="config_b_class_weight_balanced"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("evaluation_method", "5-fold CV at threshold 0.5")
        mlflow.log_param("preprocessor", "original")

        preprocessor = build_full_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=metrics,
            return_train_score=False
        )

        for metric in metrics:
            test_metric = f'test_{metric}'
            mean_score = cv_results[test_metric].mean()
            mlflow.log_metric(metric, mean_score)

        print(f"  Logged: accuracy={cv_results['test_accuracy'].mean():.4f}, "
              f"precision={cv_results['test_precision'].mean():.4f}, "
              f"recall={cv_results['test_recall'].mean():.4f}, "
              f"f1={cv_results['test_f1'].mean():.4f}")
    print()

    # Configuration C: Logistic Regression, class_weight='balanced', threshold 0.6 (Phase 7 Part 2b - best F1)
    print("Configuration C: Logistic Regression, class_weight='balanced', threshold 0.6 (Phase 7 Part 2b - best F1)")
    with mlflow.start_run(run_name="config_c_class_weight_threshold_0.6"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", 0.6)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("evaluation_method", "80/20 validation split at custom threshold")
        mlflow.log_param("preprocessor", "original")

        # Split X_train/y_train into inner train/validation for custom threshold evaluation
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        preprocessor = build_full_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ])

        # Fit on inner training set
        pipeline.fit(X_train_inner, y_train_inner)

        # Get predicted probabilities on validation set
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        # Apply custom threshold (0.6)
        y_pred = (y_proba >= 0.6).astype(int)

        # Compute metrics at threshold 0.6
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        print(f"  Logged: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
    print()

    # Historical run: Phase 7 Part 2b WITH engineered features (before preprocessor revert)
    print("Historical run: Phase 7 Part 2b WITH engineered features (before preprocessor revert)")
    with mlflow.start_run(run_name="historical_phase7_2b_WITH_engineered_features"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", 0.6)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("evaluation_method", "80/20 validation split at custom threshold")
        mlflow.log_param("preprocessor", "with_engineered_features (superseded)")
        mlflow.set_tag("superseded", "true")
        mlflow.set_tag("note", "Run before preprocessor revert - NOT the final pipeline. Features included: is_new_customer, high_risk_contract, avg_monthly_spend_ratio, num_services")

        # Log the historical metrics directly (reproduced from test_engineered_features_hypothesis.py)
        mlflow.log_metric("accuracy", 0.8012)
        mlflow.log_metric("precision", 0.6050)
        mlflow.log_metric("recall", 0.7224)
        mlflow.log_metric("f1", 0.6585)

        print(f"  Logged historical metrics: accuracy=0.8012, precision=0.6050, recall=0.7224, f1=0.6585")
    print()

    # Step 5: Log final chosen approach (Phase 8 test set metrics)
    print("Step 5: Logging final chosen approach (Phase 8 test set metrics)...")
    with mlflow.start_run(run_name="final_chosen_model"):
        mlflow.set_tag("FINAL_CHOSEN_MODEL", "true")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "None")
        mlflow.log_param("threshold", 0.3)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.log_param("preprocessor", "original")

        # Log Phase 8 test set metrics
        mlflow.log_metric("test_accuracy", 0.7495)
        mlflow.log_metric("test_precision", 0.5193)
        mlflow.log_metric("test_recall", 0.7540)
        mlflow.log_metric("test_f1", 0.6150)
        mlflow.log_metric("test_roc_auc", 0.8419)
        mlflow.log_metric("test_pr_auc", 0.6334)

        print(f"  Logged test set metrics: accuracy=0.7495, precision=0.5193, "
              f"recall=0.7540, f1=0.6150, roc_auc=0.8419, pr_auc=0.6334")
    print()

    print("=" * 70)
    print("MLflow experiment tracking complete!")
    print("=" * 70)
    print()
    print("All 4 runs have been logged to the MLflow experiment:")
    print("  - Configuration A: Baseline (no class_weight, threshold 0.5)")
    print("  - Configuration B: Class weight balanced (threshold 0.5)")
    print("  - Configuration C: Class weight balanced (threshold 0.6)")
    print("  - Final chosen model (threshold 0.3, test set metrics)")
    print()
    print("To view the MLflow dashboard, run:")
    print("  mlflow ui")
    print()
    print("Then open your browser to: http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    main()
