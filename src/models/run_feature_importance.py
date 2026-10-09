"""Feature importance analysis for Logistic Regression model.

This script extracts and visualizes the coefficients from the final chosen model
(Logistic Regression with threshold=0.3, no class_weight, no engineered features)
to understand which features the model relies on most for predictions.

IMPORTANT: These coefficients represent associations the model has learned,
not proven causal relationships. The model uses these patterns to make predictions,
but this does not mean these features cause churn.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def get_feature_names(preprocessor, X):
    """
    Extract feature names from the preprocessor.

    Returns a list of feature names in the same order as the preprocessor output.
    """
    # Fit the preprocessor to get feature names
    preprocessor.fit(X)

    # Get the column_transformer (either directly or as a step in a Pipeline)
    if hasattr(preprocessor, 'named_steps'):
        # It's a Pipeline - find the column_transformer step
        if 'column_transformer' in preprocessor.named_steps:
            column_transformer = preprocessor.named_steps['column_transformer']
        else:
            # Single-step pipeline (rare case)
            column_transformer = preprocessor
    else:
        # It's a ColumnTransformer directly
        column_transformer = preprocessor

    # Get numeric feature names (they pass through as-is)
    numeric_features = column_transformer.transformers_[0][2]

    # Get categorical feature names (one-hot encoded)
    categorical_features = column_transformer.transformers_[1][2]
    categorical_transformer = column_transformer.named_transformers_['categorical']
    onehot = categorical_transformer.named_steps['onehot']
    onehot_features = onehot.get_feature_names_out(categorical_features)

    # Combine all feature names
    all_feature_names = list(numeric_features) + list(onehot_features)

    return all_feature_names


def main():
    """Run feature importance analysis."""
    print("=" * 80)
    print("Phase 7, Part 3: Feature Importance Analysis")
    print("=" * 80)
    print()

    # Step 1: Load data and split into train/test (test set NOT used for fitting)
    print("Step 1: Loading data and splitting into train/test...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print()

    # Step 2: Build pipeline with final model configuration
    print("Step 2: Building pipeline with build_full_preprocessor() + LogisticRegression...")
    print("Configuration: max_iter=1000, random_state=42, no class_weight")
    print("Preprocessor: TotalCharges cleaning + numeric/categorical preprocessing (NO engineered features)")
    preprocessor = build_full_preprocessor()

    # Since build_full_preprocessor() returns a Pipeline, we need to add the classifier
    # by appending it to the existing pipeline steps
    pipeline = Pipeline(
        preprocessor.steps + [("classifier", LogisticRegression(max_iter=1000, random_state=42))]
    )
    print("Pipeline built successfully.")
    print()

    # Step 3: Fit pipeline on FULL training set
    print("Step 3: Fitting pipeline on FULL X_train/y_train...")
    pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")
    print()

    # Step 4: Extract feature names
    print("Step 4: Extracting feature names from preprocessor...")
    feature_names = get_feature_names(preprocessor, X_train)
    print(f"Total features after preprocessing: {len(feature_names)}")
    print()

    # Step 5: Extract coefficients
    print("Step 5: Extracting coefficients from Logistic Regression...")
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    print(f"Number of coefficients: {len(coefficients)}")
    print(f"Number of features: {len(feature_names)}")
    print()

    # Step 6: Create DataFrame with features and coefficients
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })

    # Step 7: Sort by coefficient value
    feature_importance_sorted = feature_importance.sort_values('coefficient', ascending=False)

    # Step 8: Print top 10 positive coefficients (push toward churn)
    print("=" * 80)
    print("TOP 10 FEATURES PUSHING TOWARD CHURN (POSITIVE COEFFICIENTS)")
    print("=" * 80)
    print()
    print("These features, when present or high, are associated with higher churn probability.")
    print("(Larger positive coefficient = stronger association with churn)")
    print()
    top_positive = feature_importance_sorted.head(10)
    for idx, row in top_positive.iterrows():
        print(f"{row['coefficient']:>8.4f}  {row['feature']}")
    print()

    # Step 9: Print top 10 negative coefficients (push toward staying)
    print("=" * 80)
    print("TOP 10 FEATURES PUSHING TOWARD STAYING (NEGATIVE COEFFICIENTS)")
    print("=" * 80)
    print()
    print("These features, when present or high, are associated with lower churn probability.")
    print("(More negative coefficient = stronger association with staying)")
    print()
    top_negative = feature_importance_sorted.tail(10)[::-1]  # Reverse to show most negative first
    for idx, row in top_negative.iterrows():
        print(f"{row['coefficient']:>8.4f}  {row['feature']}")
    print()

    # Step 10: Create visualization
    print("Step 10: Creating feature importance chart...")
    print()

    # Combine top 10 positive and top 10 negative
    top_features = pd.concat([top_positive, top_negative])

    # Create color list (red for positive, blue for negative)
    colors = ['red' if c > 0 else 'blue' for c in top_features['coefficient']]

    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)

    # Customize chart
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 20 Feature Coefficients - Logistic Regression\n(Positive = Associated with Churn, Negative = Associated with Staying)',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive (Churn)'),
        Patch(facecolor='blue', alpha=0.7, label='Negative (Stay)')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()

    # Save chart
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {plot_path}")
    print()

    # Step 11: Important disclaimer
    print("=" * 80)
    print("IMPORTANT DISCLAIMER")
    print("=" * 80)
    print()
    print("These coefficients represent ASSOCIATIONS the model has learned from the data,")
    print("not PROVEN CAUSAL RELATIONSHIPS.")
    print()
    print("The model uses these patterns to make predictions, but this does NOT mean that")
    print("these features cause churn. For example:")
    print("  - A feature may be correlated with another unmeasured variable that is the")
    print("    actual cause of churn")
    print("  - The relationship may be coincidental or specific to this dataset")
    print("  - Changing a feature value would not necessarily change churn behavior")
    print()
    print("Feature importance is useful for understanding what the model relies on,")
    print("but should not be used alone to make business decisions without further analysis.")
    print("=" * 80)
    print()

    print("=" * 80)
    print("NOTE: X_test and y_test were NOT used for fitting.")
    print("The model was trained on the full training set (X_train, y_train).")
    print("Test set evaluation will happen in Phase 8.")
    print("=" * 80)


if __name__ == "__main__":
    main()
