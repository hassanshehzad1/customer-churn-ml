"""Script for Phase 8, Part 2: Error Analysis on test set predictions.

This is for UNDERSTANDING and DOCUMENTATION ONLY.
We will NOT retune the model or threshold based on this analysis.
The test set evaluation from Part 1 is final and will not be reopened for tuning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_raw_data
from src.data.split import split_data
from src.features.preprocessing import build_full_preprocessor


def clean_totalcharges_series(series):
    """Clean TotalCharges Series by stripping whitespace and converting to numeric."""
    series = series.copy()
    series = series.astype(str).str.strip()
    series = series.replace("", np.nan)
    series = pd.to_numeric(series, errors="coerce")
    return series


def print_section(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_summary_stats(df, group_name, overall_stats=None):
    """Print summary statistics for a group of customers."""
    # Clean TotalCharges if it's still a string
    df_clean = df.copy()
    df_clean['TotalCharges'] = clean_totalcharges_series(df_clean['TotalCharges'])

    print(f"\n{group_name}:")
    print(f"  Number of customers: {len(df)}")
    print(f"  Mean tenure: {df_clean['tenure'].mean():.2f} months")
    print(f"  Mean MonthlyCharges: ${df_clean['MonthlyCharges'].mean():.2f}")
    print(f"  Mean TotalCharges: ${df_clean['TotalCharges'].mean():.2f}")
    print(f"  Most common Contract: {df['Contract'].mode()[0]}")
    print(f"  Most common InternetService: {df['InternetService'].mode()[0]}")
    print(f"  Most common PaymentMethod: {df['PaymentMethod'].mode()[0]}")

    if overall_stats is not None:
        print(f"\n  Difference from overall test set:")
        print(f"    Tenure: {df_clean['tenure'].mean() - overall_stats['tenure']:+.2f} months")
        print(f"    MonthlyCharges: ${df_clean['MonthlyCharges'].mean() - overall_stats['MonthlyCharges']:+.2f}")
        print(f"    TotalCharges: ${df_clean['TotalCharges'].mean() - overall_stats['TotalCharges']:+.2f}")


def print_probability_distribution(y_proba, group_name):
    """Print predicted probability distribution statistics."""
    print(f"\n{group_name} - Predicted Probability Distribution:")
    print(f"  Min: {y_proba.min():.4f}")
    print(f"  25th percentile: {np.percentile(y_proba, 25):.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    print(f"  75th percentile: {np.percentile(y_proba, 75):.4f}")
    print(f"  Max: {y_proba.max():.4f}")
    print(f"  Mean: {y_proba.mean():.4f}")
    
    # Count how many are close to threshold (within 0.05)
    threshold = 0.3
    close_to_threshold = ((y_proba >= threshold - 0.05) & (y_proba < threshold)).sum()
    far_from_threshold = (y_proba < threshold - 0.05).sum()
    
    print(f"\n  Proximity to threshold (0.3):")
    print(f"    Close calls (0.25-0.30): {close_to_threshold} ({close_to_threshold/len(y_proba)*100:.1f}%)")
    print(f"    Confident misses (<0.25): {far_from_threshold} ({far_from_threshold/len(y_proba)*100:.1f}%)")


def main():
    print_section("Phase 8, Part 2: Error Analysis on Test Set Predictions")
    print()
    print("This is for UNDERSTANDING and DOCUMENTATION ONLY.")
    print("We will NOT retune the model or threshold based on this analysis.")
    print("The test set evaluation from Part 1 is final and will not be reopened for tuning.")
    print()

    # Step 1: Load data
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print()

    # Step 2: Split data
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print()

    # Step 3: Build and fit the final pipeline (same as Part 1)
    print("Step 3: Building and fitting final pipeline (same as Part 1)...")
    preprocessor = build_full_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    print("  Model fitted successfully.")
    print()

    # Step 4: Get predicted probabilities on test set
    print("Step 4: Getting predicted probabilities on X_test...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print()

    # Step 5: Apply threshold 0.3 to get predictions
    print("Step 5: Applying threshold 0.3 to get predictions...")
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)
    print()

    # Step 6: Identify error groups
    print("Step 6: Identifying error groups...")
    
    # Create a copy of X_test for analysis (add predictions and probabilities)
    X_test_analysis = X_test.copy()
    X_test_analysis['y_true'] = y_test.values
    X_test_analysis['y_pred'] = y_pred
    X_test_analysis['y_proba'] = y_proba
    
    # Identify groups
    false_negatives = X_test_analysis[(X_test_analysis['y_true'] == 1) & (X_test_analysis['y_pred'] == 0)]
    false_positives = X_test_analysis[(X_test_analysis['y_true'] == 0) & (X_test_analysis['y_pred'] == 1)]
    true_positives = X_test_analysis[(X_test_analysis['y_true'] == 1) & (X_test_analysis['y_pred'] == 1)]
    true_negatives = X_test_analysis[(X_test_analysis['y_true'] == 0) & (X_test_analysis['y_pred'] == 0)]
    
    print(f"  False Negatives (missed churners): {len(false_negatives)}")
    print(f"  False Positives (false alarms): {len(false_positives)}")
    print(f"  True Positives (correctly caught): {len(true_positives)}")
    print(f"  True Negatives (correctly safe): {len(true_negatives)}")
    print()

    # Step 7: Compute overall test set statistics
    print("Step 7: Computing overall test set statistics...")
    # Clean TotalCharges before computing stats (it's still a string in raw X_test)
    X_test_clean = X_test.copy()
    X_test_clean['TotalCharges'] = clean_totalcharges_series(X_test_clean['TotalCharges'])
    overall_stats = {
        'tenure': X_test_clean['tenure'].mean(),
        'MonthlyCharges': X_test_clean['MonthlyCharges'].mean(),
        'TotalCharges': X_test_clean['TotalCharges'].mean()
    }
    print()

    # Step 8: Analyze False Negatives (missed churners)
    print_section("ERROR ANALYSIS: FALSE NEGATIVES (Missed Churners)")
    print("\nThese are customers who ACTUALLY churned but the model missed them.")
    print("This is the most costly error type.")
    
    print_summary_stats(false_negatives, "False Negatives", overall_stats)
    print_probability_distribution(false_negatives['y_proba'], "False Negatives")
    
    # Show some examples
    print(f"\nSample False Negative customers (first 5):")
    fn_sample = false_negatives[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
                                  'InternetService', 'PaymentMethod', 'y_proba']].head().copy()
    fn_sample['TotalCharges'] = clean_totalcharges_series(fn_sample['TotalCharges'])
    print(fn_sample.to_string())
    print()

    # Step 9: Analyze False Positives (false alarms)
    print_section("ERROR ANALYSIS: FALSE POSITIVES (False Alarms)")
    print("\nThese are customers who did NOT churn but the model flagged them as at-risk.")
    print("This is less costly - just retention effort on loyal customers.")
    
    print_summary_stats(false_positives, "False Positives", overall_stats)
    print_probability_distribution(false_positives['y_proba'], "False Positives")
    
    # Show some examples
    print(f"\nSample False Positive customers (first 5):")
    fp_sample = false_positives[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
                                  'InternetService', 'PaymentMethod', 'y_proba']].head().copy()
    fp_sample['TotalCharges'] = clean_totalcharges_series(fp_sample['TotalCharges'])
    print(fp_sample.to_string())
    print()

    # Step 10: Compare True Positives (for context)
    print_section("CONTEXT: TRUE POSITIVES (Correctly Caught Churners)")
    print("\nThese are customers who actually churned and the model caught them.")
    
    print_summary_stats(true_positives, "True Positives", overall_stats)
    print_probability_distribution(true_positives['y_proba'], "True Positives")
    print()

    # Step 11: Final insights
    print_section("ERROR ANALYSIS SUMMARY AND INSIGHTS")
    print()
    
    print("1. FALSE NEGATIVES (Missed Churners):")
    print(f"   - Count: {len(false_negatives)} customers")
    fn_clean = false_negatives.copy()
    fn_clean['TotalCharges'] = clean_totalcharges_series(fn_clean['TotalCharges'])
    fn_mean_tenure = fn_clean['tenure'].mean()
    fn_mean_monthly = fn_clean['MonthlyCharges'].mean()
    print(f"   - Average tenure: {fn_mean_tenure:.2f} months (vs {overall_stats['tenure']:.2f} overall)")
    print(f"   - Average MonthlyCharges: ${fn_mean_monthly:.2f} (vs ${overall_stats['MonthlyCharges']:.2f} overall)")
    
    if fn_mean_tenure < overall_stats['tenure']:
        print("   -> Missed churners tend to be NEWER customers (shorter tenure)")
    elif fn_mean_tenure > overall_stats['tenure']:
        print("   -> Missed churners tend to be OLDER customers (longer tenure)")
    
    fn_median_proba = np.median(false_negatives['y_proba'])
    print(f"   - Median predicted probability: {fn_median_proba:.4f}")
    if fn_median_proba >= 0.25:
        print("   -> Most missed churners are 'close calls' (probability near 0.3)")
        print("      The model was uncertain about them, not confidently wrong.")
    else:
        print("   -> Most missed churners are 'confident misses' (probability far below 0.3)")
        print("      The model confidently predicted them as safe but was wrong.")
    print()
    
    print("2. FALSE POSITIVES (False Alarms):")
    print(f"   - Count: {len(false_positives)} customers")
    fp_clean = false_positives.copy()
    fp_clean['TotalCharges'] = clean_totalcharges_series(fp_clean['TotalCharges'])
    fp_mean_tenure = fp_clean['tenure'].mean()
    fp_mean_monthly = fp_clean['MonthlyCharges'].mean()
    print(f"   - Average tenure: {fp_mean_tenure:.2f} months (vs {overall_stats['tenure']:.2f} overall)")
    print(f"   - Average MonthlyCharges: ${fp_mean_monthly:.2f} (vs ${overall_stats['MonthlyCharges']:.2f} overall)")
    
    if fp_mean_tenure < overall_stats['tenure']:
        print("   -> False alarms tend to be NEWER customers")
    elif fp_mean_tenure > overall_stats['tenure']:
        print("   -> False alarms tend to be OLDER customers")
    
    fp_median_proba = np.median(false_positives['y_proba'])
    print(f"   - Median predicted probability: {fp_median_proba:.4f}")
    print(f"   -> These are customers the model was fairly confident about (>{threshold})")
    print("      but who didn't actually churn.")
    print()
    
    print("3. KEY TAKEAWAYS:")
    print("   - These patterns help us understand the model's blind spots.")
    print("   - We will NOT retune the model based on these findings.")
    print("   - This analysis is purely for documentation and future improvement.")
    print("   - The test set evaluation from Part 1 remains final.")
    print()

    print_section("Phase 8, Part 2 COMPLETE")
    print()
    print("CRITICAL REMINDER:")
    print("  - This error analysis is for UNDERSTANDING and DOCUMENTATION ONLY.")
    print("  - We will NOT retune the model or threshold based on these findings.")
    print("  - The test set evaluation from Part 1 is final and will not be reopened.")
    print("=" * 70)


if __name__ == "__main__":
    main()
