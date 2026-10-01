"""Run data validation on the raw telco churn dataset."""

import sys
sys.path.append('.')

from src.data.load_data import load_raw_data
from src.data.validate import validate_raw_data


def main():
    """Load and validate the raw data."""
    print("Loading raw data...")
    df = load_raw_data()

    print("\nValidating raw data...")
    validation_summary = validate_raw_data(df)

    # Print validation summary in a readable format
    print("="*60)
    print("DATA VALIDATION SUMMARY")
    print("="*60)
    print(f"\nShape (rows, columns): {validation_summary['shape']}")
    print(f"\nDuplicate rows (based on customerID): {validation_summary['duplicates']}")

    print("\nMissing values per column:")
    for col, count in validation_summary['missing_per_column'].items():
        if count > 0:
            print(f"  {col}: {count}")
        else:
            print(f"  {col}: {count}")

    print(f"\nUnexpected Churn values: {validation_summary['unexpected_churn_values']}")
    print(f"\nBlank/whitespace-only TotalCharges values: {validation_summary['blank_totalcharges_count']}")
    print("="*60)


if __name__ == "__main__":
    main()
